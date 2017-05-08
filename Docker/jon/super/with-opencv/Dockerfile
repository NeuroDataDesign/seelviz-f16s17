FROM jliu118/clarityviz:super
MAINTAINER Jonathan Liu <jliu118@jhu.edu>

WORKDIR "/home"
RUN git clone https://github.com/alee156/clviz.git
WORKDIR "/home/clviz"
RUN bash opencv-docker.sh
RUN pip install -r requirements.txt

COPY ./test.py /

WORKDIR "/root"

ENTRYPOINT ["/bin/bash"]
