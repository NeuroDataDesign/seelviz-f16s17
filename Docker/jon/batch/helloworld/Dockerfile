FROM ubuntu:16.04
# FROM bids/base_fsl:5.0.9-3
MAINTAINER Jonathan Liu
RUN apt-get update
RUN apt-get install -y zip

RUN apt-get install -y python-dev wget
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py

RUN apt-get install -y git libpng-dev libfreetype6-dev pkg-config zlib1g-dev g++ vim

# RUN pip install numpy networkx>=1.11 nibabel>=2.0 dipy>=0.1 scipy boto3 matplotlib==1.5.1 plotly==1.12.1 nilearn>=0.2 sklearn>=0.0 pandas

RUN pip install boto3
#RUN mkdir ~/.aws
#RUN echo '[default]\nregion=us-east-1' >~/.aws/config

# copy over the entrypoint script
COPY ./testentrypoint.sh /
COPY ./testentrypoint.py /

# and add it as an entrypoint
# ENTRYPOINT ["/testentrypoint.sh"]
ENTRYPOINT ["python", "testentrypoint.py", "jliu118-test", "test.txt"]
