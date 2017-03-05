FROM ubuntu:16.04
MAINTAINER Seelviz

RUN apt-get update && \
    apt-get -y install build-essential git python-setuptools wget python-pip python-dev cmake python-numpy libinsighttoolkit4-dev libfftw3-dev python-matplotlib libopencv-dev python-opencv openslide-tools python-openslide
RUN apt-get -y autoremove libopencv-dev python-opencv

ENV itkVersion=4.10.0
ENV itkMinorVersion=4.10
RUN mkdir itk
WORKDIR "/root/itk"
RUN wget https://sourceforge.net/projects/itk/files/itk/${itkMinorVersion}/InsightToolkit-${itkVersion}.tar.gz
RUN tar -vxzf InsightToolkit-${itkVersion}.tar.gz
RUN mv InsightToolkit-${itkVersion} src/
RUN mkdir bin
WORKDIR "/root/itk/bin"
RUN cmake -G "Unix Makefiles" -DITK_USE_SYSTEM_FFTW=OFF -DITK_USE_FFTWD=ON -DITK_USE_FFTWF=ON -DModule_ITKReview=ON ../src
RUN make && make install

RUN yes | pip install SimpleITK ndio nibabel scikit-image pandas plotly

WORKDIR "/root"
RUN git clone https://github.com/alee156/clviz.git
WORKDIR "/root/clviz"
RUN bash opencv-docker.sh
RUN pip install -r requirements.txt

WORKDIR "/root"
RUN git clone https://github.com/jonl1096/clvizdjango.git
RUN git clone https://github.com/neurodata/ndreg.git
WORKDIR "/root/ndreg"
RUN cmake .
RUN make && make install

RUN useradd -m -s /bin/bash clv-user

WORKDIR "/root"

RUN rm -r clviz ndreg itk
RUN pip install jupyter notebook
RUN jupyter notebook --generate-config
ADD jupyter_notebook_config.py /root/.jupyter/

WORKDIR /root/clv-user