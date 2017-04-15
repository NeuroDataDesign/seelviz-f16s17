FROM ubuntu:16.04
MAINTAINER seelviz

# USER root

RUN apt-get update
# RUN apt-get -y install python-setuptools python-dev python-pip

RUN apt-get update && \
    apt-get -y install build-essential git python-setuptools wget python-pip \
	python-dev cmake python-numpy libinsighttoolkit4-dev libfftw3-dev python-matplotlib \
	libopencv-dev python-opencv openslide-tools python-openslide vim
RUN apt-get -y autoremove libopencv-dev python-opencv

ENV itkVersion=4.10.0
ENV itkMinorVersion=4.10
RUN mkdir itk
WORKDIR "/home/itk"
RUN wget https://sourceforge.net/projects/itk/files/itk/${itkMinorVersion}/InsightToolkit-${itkVersion}.tar.gz
RUN tar -vxzf InsightToolkit-${itkVersion}.tar.gz
RUN mv InsightToolkit-${itkVersion} src/
RUN mkdir bin
WORKDIR "/home/itk/bin"
RUN cmake -G "Unix Makefiles" -DITK_USE_SYSTEM_FFTW=OFF -DITK_USE_FFTWD=ON -DITK_USE_FFTWF=ON -DModule_ITKReview=ON ../src
RUN make && make install

# pip installing stuff 
RUN yes | pip install --upgrade pip \
	&& yes | pip install numpy scipy matplotlib pandas sympy nose \
	&& yes | pip install SimpleITK nibabel scikit-image plotly networkx \
	&& yes | pip install virtualenv virtualenvwrapper \
	&& Yes | pip install jupyter notebook
# NOTE: ndio used to be here, but now we install from source
# Installing Jupyter Notebook (still need to edit the config file after)

# WORKDIR "/home"
# RUN git clone https://github.com/alee156/clviz.git
# WORKDIR "/home/clviz"
# RUN bash opencv-docker.sh
# RUN pip install -r requirements.txt

WORKDIR "/home"
# RUN git clone https://github.com/jonl1096/clvizdjango.git
RUN git clone https://github.com/neurodata/ndio.git
RUN git clone https://github.com/neurodata/ndreg.git

RUN groupadd -r seelviz && useradd -r -m -g seelviz clv-user -d /home/

# RUN export WORKON_HOME=~/Envs
# RUN /bin/bash -c "source /usr/local/bin/virtualenvwrapper.sh && mkvirtualenv clv-env && workon clv-env"

# Installing ndio
WORKDIR "/home/ndio"
# RUN pip install .
RUN pip install --target /usr/local/lib/python2.7/dist-packages .

# Installing ndreg
WORKDIR "/home/ndreg"
RUN yes Y | cmake . \
	&& yes Y | make \
	&& yes Y | make install install_root=wherever
	# && yes Y | ldconfig
# && ./configure --prefix=/usr/local/lib/python2.7/dist-packages \
# RUN yes Y | ldconfig 
# DESTDIR="/usr/local/lib/python2.7/dist-packages"
RUN mkdir workspace

WORKDIR "/home"

# Deleting install files
# RUN rm -r clviz ndreg itk ndio

RUN jupyter notebook --generate-config --allow-root
# ADD jupyter_notebook_config.py /home/.jupyter/
RUN sed -i "1ic.NotebookApp.port = 8080" ~/.jupyter/jupyter_notebook_config.py \
	&& sed -i "1ic.NotebookApp.open_browser = False" ~/.jupyter/jupyter_notebook_config.py \
	&& sed -i "1ic.NotebookApp.ip = '*'" ~/.jupyter/jupyter_notebook_config.py
RUN cp -r ~/.jupyter .

# RUN chown -R clv-user /home
# RUN chmod -R 777 /home
# RUN chown -R clv-user /root/.local
# RUN chmod -R 777 /root/.local

RUN echo "alias jn='jupyter notebook --allow-root'" > ~/.bashrc

RUN chmod -R 777 /home \ 
	&& chmod 777 /usr/local/bin/virtualenvwrapper.sh
RUN mkdir clv-user
WORKDIR /root
# USER clv-user
# WORKDIR /home/clv-user

ENTRYPOINT ["/bin/bash"]
