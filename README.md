# Aplicación de Filtros de Frecuencia en Imágenes

## Descripción

Esta aplicación utiliza **Flet** para la interfaz gráfica y **OpenCV** para el procesamiento de imágenes. El programa permite aplicar diferentes filtros de frecuencia en imágenes, tanto de paso bajo (Low Pass) como de paso alto (High Pass), utilizando los siguientes métodos:

- Filtros **Butterworth**
- Filtros **Gaussianos**
- Filtros **Ideales**

La aplicación recibe una imagen en color o escala de grises, la convierte a escala de grises y permite visualizar el efecto del filtro seleccionado.

---

## Funcionalidades

1. **Selección de imágenes**:
   - Admite archivos en formato `png`, `jpg` y `jpeg`.
   - Convierte automáticamente la imagen seleccionada a **escala de grises** para su procesamiento.

2. **Filtros disponibles**:
   - **Butterworth LP** (Low Pass)
   - **Butterworth HP** (High Pass)
   - **Gaussian LP** (Low Pass)
   - **Gaussian HP** (High Pass)
   - **Ideal LP** (Low Pass)
   - **Ideal HP** (High Pass)

3. **Parámetros ajustables**:
   - **D**: Parámetro de frecuencia de corte.
   - **N**: Orden del filtro (solo para filtros Butterworth).

4. **Visualización interactiva**:
   - Muestra la imagen original en blanco y negro.
   - Presenta la imagen procesada con el filtro seleccionado.

---

## Requisitos

Para ejecutar este programa, necesitas instalar las siguientes dependencias:

- Python 3.8+
- OpenCV
- Flet
- Numpy

Puedes instalar las dependencias utilizando `pip`:

```bash
pip install opencv-python flet numpy
