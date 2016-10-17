FROM ubuntu:16.04
MAINTAINER Seelviz

WORKDIR "/"
RUN apt-get update && \
apt-get -y install build-essential && \
apt-get -y install git && \
apt-get -y install python-setuptools && \
apt-get -y install wget && \
apt-get -y install python-pip python-dev

RUN apt-get -y install cmake python-numpy libinsighttoolkit4-dev libfftw3-dev

ENV itkVersion=4.10.0
ENV itkMinorVersion=4.10
RUN mkdir itk
WORKDIR "/itk"
RUN wget https://sourceforge.net/projects/itk/files/itk/${itkMinorVersion}/InsightToolkit-${itkVersion}.tar.gz
RUN tar -vxzf InsightToolkit-${itkVersion}.tar.gz
RUN mv InsightToolkit-${itkVersion} src/
RUN mkdir bin
WORKDIR "/itk/bin"
RUN cmake -G "Unix Makefiles" -DITK_USE_SYSTEM_FFTW=OFF -DITK_USE_FFTWD=ON -DITK_USE_FFTWF=ON -DModule_ITKReview=ON ../src
RUN make && make install

WORKDIR "/"
RUN yes | pip install SimpleITK
RUN yes | pip install ndio

RUN apt-get -y install python-matplotlib
RUN yes | pip install nibabel && yes | pip install scikit-image

RUN apt-get -y install libopencv-dev python-opencv
RUN apt-get -y autoremove libopencv-dev python-opencv

WORKDIR "/home"
RUN git clone https://github.com/alee156/clviz.git
WORKDIR "/home/clviz"
RUN bash opencv-docker.sh
WORKDIR "/home"
RUN git clone https://github.com/neurodata/ndreg.git
WORKDIR "/home/ndreg"
RUN cmake .
RUN make && make install

RUN useradd -m -s /bin/bash clv-user
WORKDIR "/home/clviz"

RUN yes | pip install pandas && yes | pip install plotly
RUN pip install -r requirements.txt
RUN pip install clarityviz

WORKDIR /home/clv-user