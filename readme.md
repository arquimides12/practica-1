# 1. ¿Cómo mejorar la interfaz y la experiencia de usuario?

# # # Mejoras de interfaz (UX)

- Agregar el formulario de predicción (hoy no existe).
- Mostrar el resultado como "Sí lloverá" o "No lloverá", con su porcentaje de confianza.
- Usar campos numéricos y listas desplegables en lugar de texto libre.
- Mostrar mensajes de error simples, sin texto técnico.

# # # Mejoras de mantenibilidad
- Pipeline: para agregar un filtro solo se añade a la lista y no se toca app.py.
- config.py: los valores se cambian sin editar el código.
- services/: la web queda separada de la lógica de Machine Learning.
- tests/: cada filtro se prueba por separado.