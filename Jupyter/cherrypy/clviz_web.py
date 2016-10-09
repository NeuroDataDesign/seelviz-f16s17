import os
import os.path

import cherrypy
from cherrypy.lib import static
from cherrypy.lib.static import serve_file
import math
import clarityviz

localDir = os.path.dirname(__file__)
absDir = os.path.join(os.getcwd(), localDir)
# print absDir


def plot3d(myFile, directory, name):
    fname = name.split('.')
    out = fname[0]
    fnodes = absDir + '/' + out + 'nodes.csv'
    fedges = absDir + '/' + out + 'edges.csv'
    files = [fnodes, fedges]
    points = []
    for line in myFile:
        line = line.strip().split(',')
        points.append(str(line[0]) + "," + str(line[1]) + "," + str(line[2]))

    with open(fnodes, 'w') as nodes:
        with open(fedges, 'w') as edges:
            for ind in range(len(points)):
                temp = points[ind].strip().split(',')
                x = temp[0]
                y = temp[1]
                z = temp[2]
                radius = 25
                nodes.write("s" + str(ind + 1) + "," + str(x) + "," + str(y) + "," + str(z) + "\n")
                for index in range(ind + 1, len(points)):
                    tmp = points[index].strip().split(',')
                    distance = math.sqrt(math.pow(int(x) - int(tmp[0]), 2) + math.pow(int(y) - int(tmp[1]), 2) + math.pow(int(z) - int(tmp[2]), 2))
                    if distance < radius:
                        edgeweight = math.exp(-1 * distance)
                        edges.write("s" + str(ind + 1) + "," + "s" + str(index + 1) + "," + str(edgeweight) + "\n")

    return files


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
            <h2>Download a file</h2>
            <a href='download'>This one</a>
        </body></html>
        """

    @cherrypy.expose
    def upload(self, myFile):

        out = plot3d(myFile.file, absDir, myFile.filename)

        html = """
        <html><body>
            <h2>Ouputs</h2>
            <a href="index?directory=%s">Up</a><br />
        """ % os.path.dirname(os.path.abspath("."))
        # print os.path.dirname(os.path.abspath("."))

        for filename in out:
            absPath = os.path.abspath(filename)
            if os.path.isdir(absPath):
                link = '<a href="/index?directory=' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
                html += link
            else:
                link = '<a href="/download/?filepath=' + absPath + '">' + os.path.basename(filename) + "</a> <br />"
                html += link
        html += """</body></html>"""

        return html

    index.exposed = True


class Download:

    def index(self, filepath):
        return serve_file(filepath, "application/x-download", "attachment")

    index.exposed = True


tutconf = os.path.join(os.path.dirname('/usr/local/lib/python2.7/dist-packages/cherrypy/tutorial/'), 'tutorial.conf')
# print tutconf

if __name__ == '__main__':
    # CherryPy always starts with app.root when trying to map request URIs
    # to objects, so we need to mount a request handler root. A request
    # to '/' will be mapped to HelloWorld().index().
    root = FileDemo()
    root.download = Download()
    cherrypy.tree.mount(root)
    cherrypy.quickstart(root, config=tutconf)