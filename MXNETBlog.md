Building Deep Neural Networks in the Cloud with Azure GPU VMs, MXNet and Microsoft R Server
===========================================================================================

By Max Kaznady, Data Scientist; Richin Jain, Solution Architect; Tao Wu,
Principal Data Scientist Manager; Miguel Fierro, Data Scientist and
Andreas Argyriou, Data Scientist

**Deep learning** is a key enabler in the recent breakthroughs in
several machine learning applications. In computer vision, novel
approaches such as [deep residual
learning](https://arxiv.org/pdf/1512.03385v1.pdf) developed at Microsoft
Research helped reduce the top-5 classification error at ImageNet
competition by 47% in just one year. In speech and machine translation,
deep neural networks (DNNs) have already enabled millions of Skype users
to [communicate without language
barriers](https://www.microsoft.com/en-us/research/enabling-cross-lingual-conversations-real-time/).

Two major factors contribute to deep learning’s success: 1) availability
of large training datasets, and 2) compute acceleration that general
purpose graphics processing unit (GPU) offers. Microsoft’s Azure cloud
ecosystem, a scalable and elastic big data platform, recently introduced
advanced GPU support in its [N-Series Virtual
Machines](https://azure.microsoft.com/en-us/blog/azure-n-series-preview-availability/).
These VMs combine powerful hardware (NVIDIA Tesla K80 or M60 GPUs) with
cutting-edge, highly efficient integration technologies such as
[Discrete Device
Assignment](https://channel9.msdn.com/Shows/Azure-Friday/Leveraging-NVIDIA-GPUs-in-Azure),
bringing a new level of deep learning capability to public clouds.

This article is the first of a series of blogs that showcases deep
learning workflows on Azure. In this article, we will go over setting up
N-Series VMs on Azure with NVIDIA CUDA and cuDNN support. We use MXNet
as an example of deep learning frameworks that can run on Azure.
[MXNet](https://github.com/dmlc/mxnet) is an open-source framework for
deep neural networks with support for multiple languages and platforms
that aims to provide both execution efficiency and design flexibility.
In addition, we will also show how [Microsoft R
Server](https://www.microsoft.com/en-us/cloud-platform/r-server) can
harness the deep learning capabilities provided by MXNet and GPUs on
Azure using simple R scripts.

Preparation
===========

For this blog, we will use an NC24 VM running on Ubuntu 16.04. N-Series
VM sizes are currently under preview and available for select users; you
can register interest at http://gpu.azure.com/. In addition to the
default Ubuntu 16.04 distribution, the following libraries were used:

-   **CUDA** - CUDA8.0 RC1 (registration with NVIDIA required). In
    addition to the base package, you also need to download CUDA Patch 1
    from CUDA website. The patch adds support for gcc 5.4 as one of the
    host compilers.

-   **cuDNN** – cuDNN 5.1 (registration with NVIDIA required).

-   **Math Kernel Library** (MKL) - MKL 11.3 update 3 (registration with
    Intel required). The serial number and download link will be in
    the email.

-   **MXNet** - We used MXNet commit SHA
    f6fa98d645d2b9871e7ac5f0ad977c1e5af80738 from GitHub (which was the
    latest version of MXNet at the time)

-   **Microsoft R Server** (MRS) - Microsoft R Server 8.0.5
    (registration with Microsoft required). Alternatively, one can
    download Microsoft R Open (MRO) for Ubuntu here. Please note that
    while MRS comes with Intel MKL already bundled into the package, MRO
    requires an additional MKL installation from this link. Also, while
    MRS and MRO both rely on MKL, a separate MKL installation is
    required to build MXNet. This is because the MKL package for
    Microsoft R only contains shared libraries and not the header files
    which are needed to build external packages like MXNet.

-   **CIFAR-10 training algorithm** – test script used to validate MXNet
    installation by training a simple ResNet deep neural network on
    CIFAR-10 dataset.

-   

Installation
============

In this section, we provide step-by-step instructions to install all
components discussed earlier with their dependencies. The installation
can be completed in an hour or less. It is also important to note that
you can “copy” a configured VM for future usage, making the installation
a one-time process. Furthermore, you can create a generalized image of
the configured VM and use it in an ARM template to create similar VMs,
you can learn more about it here.

We recommend using Ubuntu version 16.04 or later, because it comes ready
with a recent Linux kernel that contains the pass-through driver needed
to recognize the GPU instances (made available to these VMs).

For installation, we assume all the packages (CUDA, cuDNN, MKL and
MXNet) are in the user’s home directory.

a.  The first step is to install the following dependencies (you can
    replace the Python installation with a local Anaconda one later if
    you want to use a different version of Python):

> sudo apt-get install -y libatlas-base-dev libopencv-dev libprotoc-dev
> python-numpy python-scipy make unzip git gcc g++ libcurl4-openssl-dev
> libssl-dev
>
> followed by update to alternatives for cc:
>
> sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 50

a.  Install downloaded CUDA driver:

    sudo ./cuda\_8.0.27\_linux.run --override

    During installation, select the following options when prompted:

-   Install NVIDIA Accelerated Graphics Driver for Linux-x86\_64
    361.77? - Yes

-   Do you want to install the OpenGL libraries? - Yes

-   Do you want to run nvidia-xconfig? – this is not necessary for
    this article.

-   Install the CUDA 8.0 Toolkit? – Yes

-   Enter Toolkit Location \[default is /usr/local/cuda-8.0\] - Select
    Default

-   -   Do you want to install a symbolic link at /usr/local/cuda? – Yes

-   Install the CUDA 8.0 Samples? *–* they are not needed for
    this article.

a.  Next, run the cuda patch 1 that you downloaded above, to support gcc
    5.4 as host compiler

> sudo ./cuda\_8.0.27.1\_linux.run
>
> Select the same options as in the previous step – default location for
> toolkit installation should be the same. Next, update alternatives for
> nvcc:
>
> sudo update-alternatives --install /usr/bin/nvcc nvcc /usr/bin/gcc 50

At this point, running the nvidia-smi command, a GPU management and
monitoring tool that is part of the CUDA package, should result in
something like the following screenshot.

> ![](media/image1.png){width="5.8125in" height="3.673176946631671in"}

a.  Install downloaded cuDNN and create a symbolic link for cudnn.h
    header file:

    tar xvzf cudnn-8.0-linux-x64-v5.1.tgz

    sudo mv cuda /usr/local/cudnn

    sudo ln -s /usr/local/cudnn/include/cudnn.h
    /usr/local/cuda/include/cudnn.h

b.  Install MKL:

    tar xvzf l\_mkl\_11.3.3.210.tgz

    sudo ./l\_mkl\_11.3.3.210/install.sh

    Follow the prompt and enter the MKL serial number that you received
    in email from intel. The default installation location is /opt/intel
    – you’ll need this for the next step.

c.  Install and build MXNet:

> First, get MXNet code from its GitHub repository (we tested the
> version with SHA f6fa98d645d2b9871e7ac5f0ad977c1e5af80738). For
> convenience, we will refer to MXNet directory path on your disk as
> MXNET\_HOME.
>
> git clone --recursive https://github.com/dmlc/mxnet
>
> cd mxnet
>
> git checkout f6fa98d645d2b9871e7ac5f0ad977c1e5af80738
>
> cp make/config.mk .
>
> Also, please note that MXNet repo has the following submodules – we
> list the SHAs for each submodule below:

-   dmlc-core: c33865feec034f1bc6ef9ec246a1ee95ac7ff148

-   mshadow: db4c01523e8d95277eae3bb52eb12260b46d6e03

-   ps-lite: 36b015ffd51c0f7062bba845f01164c0433dc6b3

    You can revert each submodule by going to its folder and running the
    same “git checkout &lt;SHA&gt;” command.

    Please note that we’re using the checkout mechanism, which means
    that you can either go back to current MXNet state after the build,
    or branch from the state and do your own work going forward.

    Next, modify the config.mk make file to use CUDA, cuDNN and MKL. You
    need to enable the flags and provide locations of the installed
    libraries:

    USE\_CUDA = 1

    USE\_CUDA\_PATH = /usr/local/cuda

    USE\_CUDNN = 1

    If MKL is to be used, USE\_BLAS and USE\_INTEL\_PATH should be set
    as follows (you can remove default “atlas” setting and replace it
    with MKL):

    USE\_BLAS = mkl

    USE\_INTEL\_PATH = /opt/intel/

    To enable distributed computing, set:

    USE\_DIST\_KVSTORE = 1

    Finally, you need to add links to CUDA and cuDNN libraries. You can
    persist those on the system by modifying /etc/environment, but since
    this is a local build, we recommend adding the following lines to
    your \~/.bashrc file instead:

    export
    LD\_LIBRARY\_PATH=/usr/local/cuda/lib64/:/usr/local/cudnn/lib64/:\$LD\_LIBRARY\_PATH

    export LIBRARY\_PATH=/usr/local/cudnn/lib64/

    Now it is time to build – you can type “bash” in the current prompt
    to apply the aforementioned changes to .bashrc or open a new
    terminal or simply re-type the above export commands into the
    current terminal.

    Next, if you want to build in parallel, use the –j option as follows
    from MXNET\_HOME:

    make –j\${nproc}

    a.  To install MRS, follow these steps:

        tar xvzf
        en\_microsoft\_r\_server\_for\_linux\_x64\_8944657.tar.gz

        cd MRS80LINUX

        sudo ./install.sh

        sudo mv /usr/lib64/microsoft-r/8.0/lib64/R/deps/libstdc++.so.6
        /tmp

        sudo mv /usr/lib64/microsoft-r/8.0/lib64/R/deps/libgomp.so.1
        /tmp

        To add MXNet library into MRS, first add the following two lines
        to /etc/ld.so.conf:

        /usr/local/cuda/lib64/

        /usr/local/cudnn/lib64/

        followed by reconfiguring dynamic linker run-time bindings:

        sudo ldconfig

> Next, make sure you’re again in the MXNET\_HOME folder:

sudo Rscript -e "install.packages('devtools', repo =
'https://cran.rstudio.com')"

cd R-package

sudo Rscript -e "install.packages(c('Rcpp', 'DiagrammeR', 'data.table',
'jsonlite', 'magrittr', 'stringr', 'roxygen2'), repos =
'https://cran.rstudio.com')"

cd ..

make rpkg

sudo R CMD INSTALL mxnet\_0.7.tar.gz

We now have a functional VM installed with MXNet, MRS and GPU. As we
suggested earlier, you can “copy” this VM for use in the future so the
installation process does not need to be repeated.

Troubleshooting
---------------

Here is some information in case you see some error messages:

1.  **Build error with** **im2rec:** If that’s the case, the easiest
    thing to do is to disable it in \$MXNET\_HOME/Makefile by commenting
    out the line “BIN += bin/im2rec”.

2.  **MKL not linking correctly**: The default root of the MKL
    installation is “/opt/intel”. If you install MKL in a different
    location, you should specify in \$MXNET\_HOME/config.mk. Note the
    path should point to the parent directory and not the MKL folder.

3.  **Library linking errors during MXNet compilation:** Make sure that
    LD\_LIBRARY\_PATH is set correctly as specified earlier (Azure GPU
    VMs come with blank LD\_LIBRARY\_PATH by default).

Test Drive
==========

Now it’s time to build some deep neural networks! Here, we use the
CIFAR-10 problem and dataset as an example. This is a 10-class
classification problem, and the dataset has 60,000 color images (6,000
images per class). We published a simple CIFAR-10 training algorithm
which can be executed from either MRS or MRO. You should first install a
few dependencies which don’t come standard with MRS:

sudo Rscript -e "install.packages('argparse', repo =
'https://cran.rstudio.com')"

Now you can run the following command from the extracted folder:

Rscript train\_resnet\_dynamic\_reload.R

You should see output which is similar to the screenshot below:

![](media/image2.png){width="6.5in" height="1.3083333333333333in"}

You can monitor GPU utilization using “watch -n 0.5 nvidia-smi” command,
which should result in something like the following (and refreshed twice
a second):

![](media/image3.png){width="6.5in" height="4.567361111111111in"}

In the screenshot above, we can see that the training is taking place on
GPU \#3.

If you are curious how to train the same model without GPU, simply
change the default training context by adding “--cpu T”, which should
produce a similar output (we also highly recommend Linux “htop” utility
for monitoring CPU usage):

![](media/image4.png){width="6.5in" height="1.1444444444444444in"}

In this case, training for 2 Epochs using CPU completes in 119.5
minutes:

![](media/image5.png){width="6.5in" height="3.1618055555555555in"}

As a comparison, training for the same 2 Epochs with GPU completes in
2.4 minutes as shown below. By using GPU, we have achieved 50x speedup
in this example.

![](media/image6.png){width="6.5in" height="3.270138888888889in"}

1.  Some More Details about Training on CPU vs GPU
    ==============================================

    Now that we have trained an MXNet model using both GPU and CPU, here
    is some more behind-the-scene information about how computation is
    done at each setup.

CPU
---

When training using CPU, Intel Math Kernel library provides great
speedup in basic linear algebra operations required for Deep Learning;
other libraries which can be used in its place are ATLAS and OpenBLAS.

Another important library which MXNet utilizes under the hood is OpenMP,
which allows multithreading of C/C++ programs without too much effort
from the developer. By adding \#pragma directives to make loops
parallel, developers can avoid managing threads explicitly.

Finally, since deep learning is commonly used in the vision domain,
OpenCV vision library is also required. This library automates most
computer vision tasks, which MXNet relies heavily on for pre-processing.

GPU
---

Convolutional operations found in deep neural networks are traditionally
very slow to execute on CPUs. GPUs are great at accelerating these types
of operations and other linear algebra routines required to train deep
neural networks.

[CUDA](https://developer.nvidia.com/cuda-zone) is the primary platform
which allows programing GPU operations from within C/C++ code on an
NVIDIA GPU. NVIDIA also provides the
[cuDNN](https://developer.nvidia.com/cudnn) library which is more
specialized for accelerating specific deep learning operations on the
GPU. Both libraries accelerate MXNet operations directly on the GPU.
Azure GPU-enabled VMs have minimal GPU virtualization overhead.

Summary
=======

In this article, we demonstrated how to quickly install and configure
MXNet on an Azure N-Series VM equipped with NVIDIA Tesla K80 GPUs. We
showed how to run MXNet training workload from Microsoft R Server using
GPU, achieving significant speedups compared to the CPU-only solution.
In the next blog, we will discuss a more comprehensive deep learning
workflow that includes accelerated training on Azure GPU VMs, scalable
scoring on HDInsight that integrates with Microsoft R Server and Apache
Spark, accessing data on Azure Data Lake Store. We will also present the
above work at the upcoming Microsoft Data Science Summit in Atlanta, GA.
Stay tuned!

### Acknowledgements

We would like to thank Qiang Kou, Tianqi Chen and other MXNet developers
and Karan Batta and Huseyin Yildiz from Microsoft Azure team for their
time and assistance in this work. We also would like to acknowledge the
use of CIFAR dataset from Alex Krizhevsky’s work in our test script.
