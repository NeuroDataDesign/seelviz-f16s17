FROM jliu118/clarityviz:super-opencv
MAINTAINER Jonathan Liu <jliu118@jhu.edu>
RUN pip install clarityviz

COPY ./execute.py /root
COPY ./execute.sh /root
RUN ["chmod", "+x", "/root/execute.sh"]
RUN mkdir output
ENTRYPOINT ["/root/execute.sh"]
CMD ["clviz-bucket", "test"]
