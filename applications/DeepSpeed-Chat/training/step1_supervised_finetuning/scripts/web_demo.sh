cd "$(dirname "${BASH_SOURCE}")";

#lsof -n -i4TCP:8081 | awk '{print $2}' | xargs kill -9

function _notify() {
    local msg=`cat /dev/stdin`
    curl -H "Accept: application/json" -X POST -d "{\"msgtype\": \"text\", \"text\": {\"content\": \"$msg\"}}" http://in.qyapi.weixin.qq.com/cgi-bin/webhook/send?key=cd1d1b5f-bbcc-46bf-85df-fc0a2d08c7b4
}
alias notify='_notify'
echo $(echo web_demo: http://`tail -n1 /etc/hosts | cut -f 1`:8081) | notify
python -m pip install -r ./requirements.txt

streamlit run --server.port=8081 ./web_demo.py


