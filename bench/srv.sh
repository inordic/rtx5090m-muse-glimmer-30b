#!/bin/bash
# srv.sh start NAME ENGINE ARGS... | srv.sh stop
# ENGINE: nf (NInferno serve) | lc (llama.cpp server binary given as LC_BIN) ; logs in /root/bench/logs/NAME.log
L=/root/bench/logs; mkdir -p $L
NF=${NF_BIN:-/opt/src/NInferno/build/apps/ninferno-serve}
LC=${LC_BIN:-/opt/llama.cpp/llama-server}
stop() {
  pkill -INT -x ninferno-serve 2>/dev/null; pkill -INT -x llama-server 2>/dev/null
  for i in $(seq 1 60); do pgrep -x ninferno-serve >/dev/null || pgrep -x llama-server >/dev/null || break; sleep 1; done
  pkill -9 -x ninferno-serve 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; sleep 2
}
if [ "$1" = stop ]; then stop; exit 0; fi
name=$2; eng=$3; shift 3
stop
if [ "$eng" = nf ]; then
  nohup $NF "$@" > $L/$name.log 2>&1 < /dev/null &
else
  nohup $LC "$@" > $L/$name.log 2>&1 < /dev/null &
fi
sleep 3   # let the new process exec under its own name before checking it
for i in $(seq 1 1200); do
  if grep -q -E "listening on|server is listening|main: server is listening" $L/$name.log; then echo "READY $name"; exit 0; fi
  if ! pgrep -x ninferno-serve >/dev/null && ! pgrep -x llama-server >/dev/null; then echo "DIED $name"; grep -i -E "error|fatal|refus|exceed" $L/$name.log | tail -5; exit 1; fi
  sleep 1
done
echo "TIMEOUT $name"; exit 1
