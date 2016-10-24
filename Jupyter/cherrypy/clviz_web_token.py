import os
import os.path

import cherrypy
from cherrypy.lib import static
from cherrypy.lib.static import serve_file

import shutil
import tempfile
import glob

from clarityviz import claritybase

localDir = os.path.dirname(__file__)
absDir = os.path.join(os.getcwd(), localDir)
# print absDir

def image_parse(filename):
    copydir = os.path.join(os.getcwd(), os.path.dirname('img/'))
    img = claritybase(filename, copydir)
    img.loadEqImg()
    img.applyLocalEq()
    img.loadGeneratedNii()
    img.calculatePoints(threshold = 0.9, sample = 0.05)
    img.savePoints()
    img.generate_plotly_html()
    img.plot3d()
    img.graphmlconvert()

class FileDemo(object):
    
    @cherrypy.expose
    def index(self, directory="."):
        img = []
        for filename in glob.glob('img/*'):
            img.append(filename)
        html = """
        <!DOCTYPE html>
        <html lang="en">

        <head>

            <meta charset="utf-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <meta name="description" content="">
            <meta name="author" content="">

            <title>ClarityViz</title>

            <!-- Bootstrap Core CSS -->
            <link href="static/vendor/bootstrap/css/bootstrap.min.css" rel="stylesheet">

            <!-- Custom Fonts -->
            <link href="static/vendor/font-awesome/css/font-awesome.min.css" rel="stylesheet" type="text/css">
            <link href='https://fonts.googleapis.com/css?family=Open+Sans:300italic,400italic,600italic,700italic,800italic,400,300,600,700,800' rel='stylesheet' type='text/css'>
            <link href='https://fonts.googleapis.com/css?family=Merriweather:400,300,300italic,400italic,700,700italic,900,900italic' rel='stylesheet' type='text/css'>

            <!-- Plugin CSS -->
            <link href="static/vendor/magnific-popup/magnific-popup.css" rel="stylesheet">

            <!-- Theme CSS -->
            <link href="static/css/creative.min.css" rel="stylesheet">

            <!-- Custom styles for this template -->
            <link href="static/css/style.css" rel="stylesheet">

            <!-- HTML5 Shim and Respond.js IE8 support of HTML5 elements and media queries -->
            <!-- WARNING: Respond.js doesn't work if you view the page via file:// -->
            <!--[if lt IE 9]>
                <script src="https://oss.maxcdn.com/libs/html5shiv/3.7.0/html5shiv.js"></script>
                <script src="https://oss.maxcdn.com/libs/respond.js/1.4.2/respond.min.js"></script>
            <![endif]-->

        </head>

        <body id="page-top">

            <nav id="mainNav" class="navbar navbar-default navbar-fixed-top">
                <div class="container-fluid">
                    <!-- Brand and toggle get grouped for better mobile display -->
                    <div class="navbar-header">
                        <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1">
                            <span class="sr-only">Toggle navigation</span> Menu <i class="fa fa-bars"></i>
                        </button>
                        <a class="navbar-brand page-scroll" href="#page-top">ClarityViz</a>
                    </div>

                    <!-- Collect the nav links, forms, and other content for toggling -->
                    <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                        <ul class="nav navbar-nav navbar-right">
                            <li>
                                <a class="page-scroll" href="https://neurodatadesign.github.io/seelviz/#Project">Project Description</a>
                            </li>
                            <li>
                                <a class="page-scroll" href="https://neurodatadesign.github.io/seelviz/#Graph">Graph Explanations</a>
                            </li>
                            <li>
                                <a class="page-scroll" href="https://neurodatadesign.github.io/seelviz/#About">About Us</a>
                            </li>
                            <li>
                                <a class="page-scroll" href="https://neurodatadesign.github.io/seelviz/#Acknowledgments">Acknowledgements</a>
                            </li>
                        </ul>
                    </div>
                    <!-- /.navbar-collapse -->
                </div>
                <!-- /.container-fluid -->
            </nav>

            <header>
                <div class="header-content">
                    <div class="header-content-inner">
                        <h1 id="homeHeading">Select File</h1>
                        <hr>
                        <!-- Columns start at 50% wide on mobile and bump up to 33.3% wide on desktop -->
                        <div class="row">
                            <div class="col-xs-6 col-md-4"></div>
                            <div class="col-xs-6 col-md-4">
                                <form action="upload" method="post" enctype="multipart/form-data">
                                    <div class="form-group">
                                        <label for="myFile">Upload a File</label>
                                        <div class="center-block"></div>
                                        <input type="file" class="form-control" id="myFile" name="myFile">
                                        <!-- <p class="help-block">Example block-level help text here.</p> -->
                                    </div>
                                    <!-- filename: <input type="file" name="myFile" /><br /> -->
                                    <input class="btn btn-default" type="submit" value="Submit">
                                </form>

                                <h2>OR</h2>

                                <form action="neurodata" method="post" enctype="multipart/form-data">
                                    <div class="form-group">
                                        <label for="myToken">Submit Token</label>
                                        <input type="text" class="form-control" id="myToken" name="myToken" placeholder="Token">
                                    </div>
                                    <!-- Token name: <input type="text" name="myToken"/><br /> -->
                                    <input class="btn btn-default" type="submit" value="Submit">
                                </form> 
                            </div>
                            <div class="col-xs-6 col-md-4"></div>
                        </div>
                    </div>
                </div>
            </header>

            <section class="bg-primary" id="about">
                <div class="container">
                    <div class="row">
                        <div class="col-lg-8 col-lg-offset-2 text-center">
                            <h2 class="section-heading">We've got what you need!</h2>
                            <hr class="light">
                            <p class="text-faded">Start Bootstrap has everything you need to get your new website up and running in no time! All of the templates and themes on Start Bootstrap are open source, free to download, and easy to use. No strings attached!</p>
                            <a href="#services" class="page-scroll btn btn-default btn-xl sr-button">Get Started!</a>
                        </div>
                    </div>
                </div>
            </section>

            <section id="contact">
                <div class="container">
                    <div class="row">
                        <div class="col-lg-8 col-lg-offset-2 text-center">
                            <h2 class="section-heading">Acknowledgements</h2>
                            <hr class="primary">
                            <p>Ready to start your next project with us? That's great! Give us a call or send us an email and we will get back to you as soon as possible!</p>
                        </div>
                        <div class="col-lg-4 col-lg-offset-2 text-center">
                            <i class="fa fa-phone fa-3x sr-contact"></i>
                            <p>123-456-6789</p>
                        </div>
                        <div class="col-lg-4 text-center">
                            <i class="fa fa-envelope-o fa-3x sr-contact"></i>
                            <p><a href="mailto:your-email@your-domain.com">feedback@startbootstrap.com</a></p>
                        </div>
                    </div>
                </div>
            </section>

            <!-- jQuery -->
            <script src="vendor/jquery/jquery.min.js"></script>

            <!-- Bootstrap Core JavaScript -->
            <script src="vendor/bootstrap/js/bootstrap.min.js"></script>

            <!-- Plugin JavaScript -->
            <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-easing/1.3/jquery.easing.min.js"></script>
            <script src="vendor/scrollreveal/scrollreveal.min.js"></script>
            <script src="vendor/magnific-popup/jquery.magnific-popup.min.js"></script>

            <!-- Theme JavaScript -->
            <script src="js/creative.min.js"></script>

        </body>

        </html>

        """
        return html

    @cherrypy.expose
    def neurodata(self, myToken):
        fzip = shutil.make_archive(myToken, 'zip', myToken)
        fzip_abs = os.path.abspath(fzip)
        image_parse(myToken)
        html = """
        <html><body>
            <h2>Ouputs</h2>
        """
#            <a href="index?directory=%s">Up</a><br />
#        """ % os.path.dirname(os.path.abspath("."))
        # print os.path.dirname(os.path.abspath("."))

        # for filename in out:
        plotly = []
        for filename in glob.glob(myToken + '/*'):
            absPath = os.path.abspath(filename)
            if os.path.isdir(absPath):
                link = '<a href="/index?directory=' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
                html += link
            else:
                if filename.endswith('html'):
                    plotly.append(filename)
                link = '<a href="/download/?filepath=' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
                html += link

        for plot in plotly:
            absPath = os.path.abspath(plot)
            html += """
              <form action="plotly" method="get">
                <input type="text" value=""" + '"' + absPath + '" name="plot" ' + """/>
                <button type="submit">View """ + os.path.basename(plot) + """</button>
              </form>"""
            # html += '<a href="file:///' + '//' + absPath + '">' + "View Plotly graph</a> <br />"

        html += '<a href="/download/?filepath=' + fzip_abs + '">' + myToken + '.zip' + "</a> <br />"
        html += """</body></html>"""

        return html   

    @cherrypy.expose
    def upload(self, myFile):
#        destination = os.path.join('local/')
#        print destination
#        with open(destination + myFile.filename, 'wb') as f:
#            shutil.copyfileobj(cherrypy.request.body, f)
        # print os.path.dirname(os.path.realpath(myFile.file))
        copy = 'local/' + myFile.filename
        print copy
        token = myFile.filename.split('.')[:-1]

        with open(copy, 'wb') as fcopy:
            while True:
                data = myFile.file.read(8192)
                if not data:
                    break
                fcopy.write(data)

        copydir = os.path.join(os.getcwd(), os.path.dirname('local/'))
        print copydir
        csv = claritybase.claritybase(token, copydir)
        # out = plot3d(myFile.file, absDir, myFile.filename)
        # out = plot3d('local/', myFile.filename)
        csv.loadInitCsv(copydir + '/' + myFile.filename)
        csv.plot3d()
        csv.savePoints()
        csv.generate_plotly_html()
        csv.graphmlconvert()
        fzip = shutil.make_archive(token, 'zip', token)
        fzip_abs = os.path.abspath(fzip)

        html = """
        <html><body>
            <h2>Ouputs</h2>
        """
#            <a href="index?directory=%s">Up</a><br />
#        """ % os.path.dirname(os.path.abspath("."))
        # print os.path.dirname(os.path.abspath("."))
        # for filename in out:
        plotly = []
        for filename in glob.glob(token + '/*'):
            absPath = os.path.abspath(filename)
            if os.path.isdir(absPath):
                link = '<a href="/index?directory=' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
                html += link
            else:
                if filename.endswith('html'):
                    plotly.append(filename)
                link = '<a href="/download/?filepath=' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
                html += link

        for plot in plotly:
            absPath = os.path.abspath(plot)
            html += """
              <form action="plotly" method="get">
                <input type="text" value=""" + '"' + absPath + '" name="plot" ' + """/>
                <button type="submit">View """ + os.path.basename(plot) + """</button>
              </form>"""
            # html += '<a href="file:///' + '//' + absPath + '">' + "View Plotly graph</a> <br />"

        html += '<a href="/download/?filepath=' + fzip_abs + '">' + token + '.zip' + "</a> <br />"
        html += """</body></html>"""

        return html

    @cherrypy.expose
    def plotly(self, plot="test/testplotly.html"):
        return file(plot)
    # index.exposed = True


class Download:

    @cherrypy.expose
    def index(self, filepath):
        return serve_file(filepath, "application/x-download", "attachment")

    # index.exposed = True


tutconf = os.path.join(os.path.dirname('/usr/local/lib/python2.7/site-packages/cherrypy/tutorial/'), 'tutorial.conf')
# print tutconf


if __name__ == '__main__':
    # CherryPy always starts with app.root when trying to map request URIs
    # to objects, so we need to mount a request handler root. A request
    # to '/' will be mapped to HelloWorld().index().

    # On Startup
    current_dir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
    # NEED TO SET ABSOLUTE PATHS
    config = {
        'global': {
            'environment': 'production',
            'log.screen': True,
            'server.socket_host': '127.0.0.1',
            'server.socket_port': 8080,
            'server.thread_pool': 10,
            'engine.autoreload_on': True,
            'log.error_file': os.path.join(current_dir, 'errors.log'),
            'log.access_file': os.path.join(current_dir, 'access.log'),
        },
        '/':{
            'tools.staticdir.root' : current_dir,
        },
        '/static':{
            'tools.staticdir.on' : True,
            'tools.staticdir.dir' : 'static',
            'staticFilter.on': True,
            'staticFilter.dir': '/Users/Kepler/code/seelviz/Jupyter/cherrypy/static'
        },
    }
# cherrypy.quickstart(Root(), '/', config)
    root = FileDemo()
    root.download = Download()
    cherrypy.tree.mount(root)
    cherrypy.quickstart(root, '/', config=config)
    # cherrypy.quickstart(root, config=tutconf)
