FROM ubuntu:16.04
MAINTAINER Jonathan Liu

RUN apt-get update
RUN apt-get install -y zip
RUN apt-get install -y python-dev wget
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN apt-get install -y git libpng-dev libfreetype6-dev pkg-config zlib1g-dev g++ vim

RUN pip install Django
RUN yes | pip install --upgrade awscli
RUN yes | pip install boto3

WORKDIR "/root"

RUN git clone https://github.com/jonl1096/clvizdjango.git

WORKDIR "/root/clvizdjango/seelviz/"
COPY ./accessKeys.csv /root/clvizdjango/seelviz/

ENTRYPOINT ["python", "manage.py", "runserver", "0.0.0.0:8070"]
