# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import time
import torch
import torch.nn.functional as F
import sys
import os
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pandas as pd
from alpaca_farm import torch_ops
from utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank=rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3

        # Those value can be changed
        self.kl_ctl = 0.2
        #self.kl_ctl = 0.1
        self.vf_coef = 0.1
        self.clip_reward_value = 1
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        #self.lam = 0.95
        self.lam = 1.0
        self.prompt_answer = args.prompt_answer

    def _generate_sequence(self, prompts, mask):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        import os
        if 'RANK' not in os.environ:
            print('set rank')
            os.environ['RANK'] = self.args.global_rank

        if self.actor_model.model.config.model_type == "llama":
            kwargs = dict(do_sample=False)
        else:
            kwargs = dict()
        with torch.no_grad():
            #print('prompts', prompts)
            seq = self.actor_model.module.generate(prompts,
            #seq = self.ref_model.module.generate(prompts,
                attention_mask=mask,
                max_length=max_min_length,
                #min_length=max_min_length,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled,
                top_p=1.0,
                top_k=0,
                temperature=1.0,
                **kwargs,
            )
            prompt_result = self.tokenizer.batch_decode(prompts,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)


            print_rank_0('prompt_result', prompt_result, rank=self.args.global_rank)
            print_rank_0('prompts', prompts, self.args.global_rank, rank=prompts.shape)
            print_rank_0('seq', seq, self.args.global_rank, rank=seq.shape)
            result = self.tokenizer.batch_decode(seq,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            print_rank_0('result', result, rank=self.args.global_rank)

            device = seq.device
            if self.prompt_answer:
                seq = self.tokenizer(
                    result,
                    max_length=seq.shape[1],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt")['input_ids'].to(device)
            else:
                enable_prompt_pad_answer_pad = False
                # answer pad pad
                self.tokenizer.padding_side = 'right'
                seq = self.tokenizer(
                    [e.split('Assistant: ')[-1] for e in result],
                    max_length=self.max_answer_seq_len + 1,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt")['input_ids'][:, 1:].to(device)
                if enable_prompt_pad_answer_pad:
                    # shift to right pad: pad prompt
                    roll_list = []
                    for a, b in zip(prompts.cpu(), (mask.sum(1) - prompts.shape[1]).cpu().numpy()):
                        roll_list.append(torch.roll(a, b, dims=0))
                    prompts = torch.stack(roll_list, axis=0).to(device)
                seq = torch.cat([prompts, seq], dim=1)
            self.tokenizer.padding_side = 'left'
            print_rank_0('after_seq: ' + str(seq), rank=self.args.global_rank)

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        ans = seq[:, prompt_length:]
        self.prompt_length = prompt_length
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        if False:
            out_seq = []
            for i in range(batch_size):
                if valid_ans_len[
                        i] <= 1:  # if the answer is shorter than 1 token, drop it
                    continue
                else:
                    out_seq.append(seq[i:i + 1])
            out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim
        else:
            out_seq = seq

        return {
            "seq": out_seq, 
        }

    def generate_experience(self, prompts, mask, step_idx=0):
        print_rank_0(prompts, rank=self.args.global_rank)
        print_rank_0(mask, rank=self.args.global_rank)
        self.eval()
        print_rank_0("Time before generate_seq: {}".format(int(time.time())), rank=self.args.global_rank)
        gen_output = self._generate_sequence(prompts, mask)
        seq = gen_output["seq"]
        print_rank_0("Time after generate_seq: {}".format(int(time.time())), rank=self.args.global_rank)

        pad_token_id = self.tokenizer.pad_token_id
        print_rank_0('gen exp prompt' + str(prompts), rank=self.args.global_rank)
        print_rank_0('gen exp mask' + str(mask), rank=self.args.global_rank)
        attention_mask = seq.not_equal(pad_token_id).long()
        print_rank_0('gen exp attention_mask' + str(attention_mask), rank=self.args.global_rank)

        with torch.no_grad():
            print_rank_0("Time before actor_model: {}".format(int(time.time())), rank=self.args.global_rank)
            output = self.actor_model(seq, attention_mask=attention_mask)
            print_rank_0("Time before ref_model: {}".format(int(time.time())), rank=self.args.global_rank)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            print_rank_0("Time before reward_model: {}".format(int(time.time())), rank=self.args.global_rank)
            if self.prompt_answer:
                reward_score = self.reward_model.forward_value(
                    seq, attention_mask,
                    prompt_length=seq.shape[1])['chosen_end_scores'].detach(
                    )
            else:
                reward_score = self.reward_model.forward_value(
                    seq, attention_mask,
                    prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                    )

            if self.args.use_scale_reward:
                reward_score = (reward_score - self.args.scale_reward_mean) / self.args.scale_reward_std
                reward_score = torch.clamp(reward_score, min=-1.0, max=1.0)

            print_rank_0("Time before critic_model: {}".format(int(time.time())), rank=self.args.global_rank)
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]
            if self.args.use_scale_value:
                values = (values - self.args.scale_value_mean) / self.args.scale_value_std
                values = torch.clamp(values, min=-1.0, max=1.0)
 
            print_rank_0("Time after critic_model: {}".format(int(time.time())), rank=self.args.global_rank)
        self.train()

        logits = output.logits
        logits_ref = output_ref.logits

        if self.args.global_rank == 0:
            rollouts_to_disk = {
                k: self.tokenizer.batch_decode(
                    tensor, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                for k, tensor in [
                    ('prompt', prompts),
                    ('seq', seq),
                ]
            }
            rollouts_to_disk = pd.DataFrame(rollouts_to_disk).to_dict(orient="records")
            from alpaca_farm.utils import jdump
            jdump(rollouts_to_disk, os.path.join(self.args.output_dir, "rollouts", f"step_{step_idx}.json"))

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask,
        }

    def _is_correct(self, ans):
        ans = ans.strip()
        q = ans.split('Human: ')[-1].split('Assistant:')[0].strip()
        true = eval(q)
        p = ans.split('Human: ')[-1].split('Assistant:')[1].strip()
        pred = int(p)
        return int(true == pred)

    def compute_rewards(self, prompts, seq, log_probs, ref_log_probs, reward_score,
                        action_mask):

        print_rank_0('prompt' + str(prompts), rank=self.args.global_rank)
        for e in seq:
            e = self.tokenizer.convert_ids_to_tokens(e)
            print_rank_0('seq_e' + str(e), rank=self.args.global_rank)

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        #kl_divergence_estimate = -self.kl_ctl * torch.clamp(log_probs - ref_log_probs, min=0.0)

        rewards = kl_divergence_estimate.clone()
        if self.prompt_answer:
            # left padding
            start = 0
            ends = 0
        else:
            start = prompts.shape[1] - 1
            ends = start + action_mask[:, start:].sum(1) + 1
        print_rank_0('reward_score' + str(reward_score), rank=self.args.global_rank)
        print_rank_0('start' + str(start), rank=self.args.global_rank)
        print_rank_0('ends' + str(ends), rank=self.args.global_rank)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        if self.prompt_answer:
            rewards[list(range(reward_score.size(0))), -1] += reward_clip
            #for j in range(batch_size):
            #    rewards[j, start[j]:][-1] += reward_clip[j]
        else:
            #rewards[list(range(reward_score.size(0))), ends - 1] += reward_clip
            for j in range(batch_size):
                rewards[j, start:ends[j]][-1] += reward_clip[j]

        print_rank_0('reward' + str(rewards), rank=self.args.global_rank)

        kl = torch.clamp(log_probs - ref_log_probs, min=0.0)
        non_score_rewards = -self.kl_ctl * kl
        shaped_rewards = non_score_rewards.clone()
 
        return dict(
            shaped_rewards=shaped_rewards,
            non_score_rewards=non_score_rewards,
            kl=kl,
            rewards=rewards,
        )

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        if self.args.use_manual_reward:
            result = self.tokenizer.batch_decode(seq,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            try:
                device = seq.device
                correct = torch.tensor([self._is_correct(res) for res in result], device=device)
                correct_all_reduce = correct.clone()
                torch.distributed.all_reduce(correct_all_reduce, op=torch.distributed.ReduceOp.SUM)
                print_rank_0('correct_all_reduce_list: ' + str(correct_all_reduce), rank=self.args.global_rank)
                print_rank_0('correct: {}'.format(correct_all_reduce.sum()), rank=self.args.global_rank)
                reward_score = correct * 1.0 / ( torch.distributed.get_world_size() * seq.shape[0] )
                inputs['rewards'] = reward_score
            except Exception as e:
                print_rank_0("error:" + str(e), rank=self.args.global_rank)

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]
        print_rank_0('attention_mask' + str(attention_mask), rank=self.args.global_rank)

        old_values = values
        with torch.no_grad():
            rollouts = self.compute_rewards(prompts, seq, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            old_rewards = rollouts['rewards']
            if self.prompt_answer:
                pass
            else:
                ends = start + action_mask[:, start:].sum(1) + 1
                # we need to zero out the reward and value after the end of the conversation
                # otherwise the advantage/return will be wrong
                for i in range(old_rewards.shape[0]):
                    old_rewards[i, ends[i]:] = 0
                    old_values[i, ends[i]:] = 0
            print_rank_0('old_reward' + str(old_rewards), rank=self.args.global_rank)
            print_rank_0('old_value' + str(old_values), rank=self.args.global_rank)
            if self.prompt_answer:
                advantages, returns = self.get_advantages_and_returns(
                    old_values, old_rewards, start=0)
            else:
                advantages, returns = self.get_advantages_and_returns(
                    old_values, old_rewards, start)
            print_rank_0('advantages' + str(advantages), rank=self.args.global_rank)
            print_rank_0('advantages' + str(advantages.shape), rank=self.args.global_rank)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        #new_mask = 
        if self.prompt_answer:
            #from alpaca_farm import torch_ops
            #torch_ops.left_pad(input_ids, (2, 6), value=2)
            device = prompts.device
            padding = torch.ones(prompts.shape[0], action_mask.shape[1] - prompts.shape[1]).long() * self.tokenizer.pad_token_id
            prompts_pad = torch.cat([padding, prompts.cpu()], dim=1)
            prompts_mask = prompts_pad.not_equal(self.tokenizer.pad_token_id).long()
            to_be_shift = torch.logical_xor(action_mask.cpu(), prompts_mask.cpu()).long()
            tmp_arr = [torch.roll(a, [b], dims=0) for a, b in zip(to_be_shift, prompts_mask.cpu().sum(1))]
            new_mask = torch.stack(tmp_arr, dim=0).to(device)
            actor_ret = self.actor_loss_fn(actor_log_prob, log_probs, advantages, new_mask)
        else:
            actor_ret = self.actor_loss_fn(actor_log_prob[:, start:],
                                            log_probs[:, start:], advantages,
                                            action_mask[:, start:])
        actor_loss = actor_ret['pg_loss']
        self.actor_model.backward(actor_loss)
        self.actor_model.step()
        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        if self.prompt_answer:
            crit_ret = self.critic_loss_fn(value, old_values, returns, new_mask)
        else:
            crit_ret = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                           start:],
                                              returns, action_mask[:, start:])
        critic_loss = crit_ret["critic_loss"]

        self.critic_model.backward(critic_loss * self.vf_coef)
        self.critic_model.step()


        # Stats
        stats = crit_ret
        stats['policy'] = actor_ret
        stats['loss'] = dict(
            policy=actor_loss, 
            value=critic_loss,
        )
        from alpaca_farm import common
        train_stats = common.flatten_dict(stats, sep="/", postprocess_fn=lambda x: x.detach())

        kl = rollouts["kl"]
        kl_sum_seq, kl_avg_seq = kl.sum(dim=1).mean(dim=0), kl.mean()
        shaped_rewards = rollouts["shaped_rewards"].sum(dim=1).mean(dim=0)
        non_score_rewards = rollouts["non_score_rewards"].sum(dim=1).mean(dim=0)
        rewards = rollouts["rewards"].mean(dim=0)
        stats = {
            f"objective/kl_sum_seq": kl_sum_seq,
            f"objective/kl_avg_seq": kl_avg_seq,
            f"objective/shaped_rewards": shaped_rewards,
            f"objective/non_score_rewards": non_score_rewards,
            f"objective/rewards": rewards,  # Original model reward.
        }
        for k, v in train_stats.items():
            stats[f"ppo/{k}"] = v.mean(dim=0)
        #stats = {key: value.item() if torch.is_tensor(value) else value for key, value in stats.items()}
 
        for k in stats.keys():
            try:
                v = stats[k]
                if torch.is_tensor(v):
                    v = v.item()
                stats[k] = v
                line = "{}_step: {}".format(k, str(stats[k]))
                print_rank_0(line, rank=self.args.global_rank)
            except:
                print('fail convert', k, v)
                pass

        return actor_loss, critic_loss

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        print_rank_0('xor_mask' + str(mask), rank=self.args.global_rank)
        log_ratio = (logprobs - old_logprobs) * mask
        print_rank_0('log_ratio' + str(log_ratio), rank=self.args.global_rank)
        ratio = torch.exp(log_ratio)
        print_rank_0('ratio' + str(ratio), rank=self.args.global_rank)
        pg_loss1 = -advantages * ratio * mask
        print_rank_0('pg_loss1' + str(pg_loss1), rank=self.args.global_rank)
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange) * mask
        print_rank_0('pg_loss2' + str(pg_loss2), rank=self.args.global_rank)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()

        approxkl = 0.5 * ((logprobs - old_logprobs) ** 2.0).mean()
        pg_clipfrac = (pg_loss2 > pg_loss1).to(torch.get_default_dtype()).mean()  # noqa
        return dict(
            pg_loss=pg_loss,
            approxkl=approxkl,
            clipfrac=pg_clipfrac,
        )

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        print_rank_0('values_clipped' + str(values_clipped), rank=self.args.global_rank)
        vf_loss1 = (values - returns)**2 * mask
        print_rank_0('vf_loss1' + str(vf_loss1), rank=self.args.global_rank)
        vf_loss2 = (values_clipped - returns)**2 * mask
        print_rank_0('vf_loss2' + str(vf_loss2), rank=self.args.global_rank)
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()

        # stat
        vpred = values
        vf_clipfrac = (vf_loss2 > vf_loss2).to(torch.get_default_dtype()).mean()
        value_mean, value_var = old_values.mean(), old_values.var(unbiased=False)
        return_mean, return_var = returns.mean(), returns.var(unbiased=False)
        return dict(
            critic_loss=vf_loss,
            returns=dict(mean=return_mean, var=return_var),
            val=dict(
                vpred=vpred.mean(),
                error=((vpred - returns) ** 2).mean(),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            )
        )

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        max_len = self.args.max_prompt_seq_len + self.args.max_answer_seq_len
        if self.args.use_whiten:
            rewards = torch_ops.whiten(rewards, shift_mean=False, max_len=max_len)
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        if self.args.use_whiten:
            advantages = torch_ops.whiten(advantages, shift_mean=True, max_len=max_len)
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
