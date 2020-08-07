# video_server.py
from flask import Flask, render_template, Response
import object_tracker as ot
app = Flask(__name__)

@app.route("/")
def index():
    # return the rendered template
	return render_template("index.html")

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(ot.streamVideo(),mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
	app.run(host="localhost", port="5019", debug=True,
			threaded=True, use_reloader=False)

