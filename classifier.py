import builtins
import flask

app = flask.Flask(__name__)

if __name__ == "__main__":
    builtins.app = app
    import web
    app.run()
