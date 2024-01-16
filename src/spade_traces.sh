#!/bin/sh

# exec in docker container to enable tracing in space state machines
sed -i 's%Exception running state {}: {}".format(self, e))%Exception running state {}: {}".format(self, e), exc_info=True)%' /usr/local/lib/python3.10/dist-packages/spade/behaviour.py 