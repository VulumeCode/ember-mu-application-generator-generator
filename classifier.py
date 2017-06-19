import builtins
import flask

app = flask.Flask(__name__)

if __name__ == "__main__":
    builtins.app = app
    import web
    builtins.databaseURL = "http://sem-eurod01.tenforce.com:8890/sparql"
    app.run()
