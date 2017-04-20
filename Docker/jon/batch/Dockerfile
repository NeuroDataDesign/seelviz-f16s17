FROM jliu118/clarityviz:super-opencv
MAINTAINER Jonathan Liu <jliu118@jhu.edu>

RUN pip install -U scikit-learn \
	&& pip install boto3

RUN pip install clarityviz

COPY ./execute.py /

ENTRYPOINT ["python", "execute.py", "jliu118-test", "arguments.txt"]
