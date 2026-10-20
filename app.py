from flask import Flask, render_template, request, flash, redirect, url_for

from config import Config
from services import TrainingService, PredictionService, UserError


def create_app(config=Config):
    # Configuración de Flask
    app = Flask(
        __name__,
        static_folder="public",
        static_url_path="/public"
    )
    app.config.from_object(config)

    # La web solo habla con los servicios; la lógica de ML vive en services/
    training = TrainingService(config)
    prediction = PredictionService(config)

    def render(results=None, prediction_result=None, show_predict=False, status=200):
        has_model = prediction.has_model()
        show_predict = has_model and (show_predict or results is not None)
        return render_template(
            "index.html",
            results=results,
            has_model=has_model,
            fields=prediction.form_fields() if show_predict else None,
            prediction_result=prediction_result,
        ), status

    @app.route("/", methods=["GET"])
    def index():
        return render()

    @app.route("/train", methods=["POST"])
    def train():
        try:
            results = training.train(request.files.get("dataset"))
        except UserError as exc:
            flash(str(exc), "error")
            return render(status=400)
        return render(results=results)

    @app.route("/predict", methods=["GET"])
    def predict_form():
        if not prediction.has_model():
            flash("Primero entrena un modelo subiendo un archivo CSV.", "error")
            return redirect(url_for("index"))
        return render(results=prediction.last_results(), show_predict=True)

    @app.route("/predict", methods=["POST"])
    def predict():
        if not prediction.has_model():
            flash("Primero entrena un modelo subiendo un archivo CSV.", "error")
            return redirect(url_for("index"))
        try:
            result = prediction.predict(request.form)
        except UserError as exc:
            flash(str(exc), "error")
            return render(results=prediction.last_results(),
                          prediction_result={"inputs": request.form.to_dict()},
                          show_predict=True, status=400)
        return render(results=prediction.last_results(), prediction_result=result,
                      show_predict=True)

    @app.errorhandler(413)
    def too_large(_):
        flash(f"El archivo es demasiado grande. El máximo es {config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB.", "error")
        return render(status=413)

    @app.errorhandler(500)
    def server_error(_):
        flash("Ocurrió un problema inesperado. Inténtalo de nuevo.", "error")
        return render(status=500)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
