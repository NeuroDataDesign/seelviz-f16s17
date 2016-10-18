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

class FileDemo(object):

    @cherrypy.expose
    def index(self, directory="."):
        return """
        <html><body>
            <h2>Upload a file</h2>
            <form action="upload" method="post" enctype="multipart/form-data">
            filename: <input type="file" name="myFile" /><br />
            <input type="submit" />
            </form>

          <form action="neurodata" method="post" enctype="multipart/form-data">
          Token name: <input type="text" name="myToken"/><br />
          <input type="submit" />
          </form>
            
        </body></html>
        """
    
    @cherrypy.expose
    def neurodata(self, myToken):
        csv = claritybase(myToken, "./")
        # out = plot3d(myFile.file, absDir, myFile.filename)
        # out = plot3d('local/', myFile.filename)
        csv.loadInitCsv('/Users/albert/seelviz/csv/' + myToken + "localeq.csv")
        csv.plot3d()
        csv.savePoints()
        csv.generate_plotly_html()
        csv.graphmlconvert()
        fzip = shutil.make_archive(myToken, 'zip', myToken)
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
    def upload(self, myFile):
#        destination = os.path.join('local/')
#        print destination
#        with open(destination + myFile.filename, 'wb') as f:
#            shutil.copyfileobj(cherrypy.request.body, f)
        # print os.path.dirname(os.path.realpath(myFile.file))
        copy = 'local/' + myFile.filename
        print copy
        token = myFile.filename.split('.')[0]

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
    root = FileDemo()
    root.download = Download()
    cherrypy.tree.mount(root)
    cherrypy.quickstart(root, config=tutconf)
