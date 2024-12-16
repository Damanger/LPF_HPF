import flet as ft
import cv2 as cv
import numpy as np
import math
import tempfile

# Función para calcular el radio
def calculateRadius(centre, i, j):
    return max(1, math.sqrt(math.pow((i - centre[0]), 2) + math.pow((j - centre[1]), 2)))

# Crear filtros
def create_butterworth_low_pass_filter(width, height, d, n):
    lp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)
    for i in range(0, lp_filter.shape[1]):
        for j in range(0, lp_filter.shape[0]):
            radius = calculateRadius(centre, i, j)
            lp_filter[j, i] = 1 / (1 + math.pow((radius / d), (2 * n)))
    return lp_filter

def butterworth_high_pass_filter(width, height, d, n):
    return 1 - create_butterworth_low_pass_filter(width, height, d, n)

def create_gaussian_low_pass_filter(width, height, d):
    lp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)
    for i in range(0, lp_filter.shape[1]):
        for j in range(0, lp_filter.shape[0]):
            radius = calculateRadius(centre, i, j)
            lp_filter[j, i] = math.exp(-math.pow((radius / d), 2))
    return lp_filter

def gaussian_high_pass_filter(width, height, d):
    return 1 - create_gaussian_low_pass_filter(width, height, d)

def create_ideal_low_pass_filter(width, height, d):
    lp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)
    for i in range(0, lp_filter.shape[1]):
        for j in range(0, lp_filter.shape[0]):
            radius = calculateRadius(centre, i, j)
            if radius <= d:
                lp_filter[j, i] = 1
    return lp_filter

def ideal_high_pass_filter(width, height, d):
    return 1 - create_ideal_low_pass_filter(width, height, d)

# Aplicar filtro
def apply_filter(image, filter_function, d, n=None):
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    height, width = image.shape
    if n is not None:
        filter_mask = filter_function(width, height, d, n)
    else:
        filter_mask = filter_function(width, height, d)
    filtered_dft = dft_shift * filter_mask
    f_ishift = np.fft.ifftshift(filtered_dft)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalizar la imagen al rango [0, 255]
    img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX)
    img_back = np.uint8(img_back)
    return img_back

# Función principal de Flet
def main(page: ft.Page):
    page.window_width = 800
    page.window_height = 950
    page.title = "Filtros de Frecuencia"
    page.theme_mode = "dark"
    
    # Variables de la interfaz
    image_display = ft.Image(width=300, height=300)
    processed_image_display = ft.Image(width=300, height=300)
    selected_image_path = None

    # Sliders
    d_slider = ft.Slider(min=10, max=100, value=30, label=lambda value: f"Valor: {value:.0f}", on_change=lambda e: (update_filter(), page.update()))
    n_slider = ft.Slider(min=1, max=10, value=2, label=lambda value: f"Valor: {value:.0f}", on_change=lambda e: (update_filter(), page.update()))

    # Textos explicativos debajo de los sliders
    d_text = ft.Text(
        """
        Valores bajos de D:
        ● En un filtro LP, solo se permiten las frecuencias muy bajas, generando un efecto más borroso en la imagen.
        ● En un filtro HP, se permite solo el contorno más detallado, filtrando casi todo el resto.
        Valores altos de D:
        ● En un filtro LP, se permiten frecuencias más altas, manteniendo más detalles en la imagen.
        ● En un filtro HP, se bloquean solo las frecuencias más bajas, dejando más detalles del contorno.
        """
    )

    n_text = ft.Text(
        """
        N es un parámetro que solo se aplica a los filtros Butterworth (de paso bajo y alto).
        Representa el orden del filtro, que afecta la pendiente de la transición entre las frecuencias permitidas y
        bloqueadas.
        """
    )

    # Mostrar sliders condicionalmente
    sliders_column = ft.Column(controls=[], visible=False, spacing=15)

    def toggle_sliders(filter_name):
        # Etiquetas dinámicas
        d_label = ft.Text(f"D: {int(d_slider.value)}")
        n_label = ft.Text(f"N: {int(n_slider.value)}")

        # Actualizar D
        def update_d(e):
            d_label.value = f"D: {int(d_slider.value)}"
            update_filter()  # Actualizar filtro
            page.update()

        # Actualizar N
        def update_n(e):
            n_label.value = f"N: {int(n_slider.value)}"
            update_filter()  # Actualizar filtro
            page.update()

        # Configurar eventos dinámicos de sliders
        d_slider.on_change = update_d
        n_slider.on_change = update_n

        # Configurar sliders según el filtro seleccionado
        if "Butterworth" in filter_name:
            sliders_column.controls = [
                d_label, d_slider, d_text,
                n_label, n_slider, n_text
            ]
            sliders_column.visible = True
        elif "LP" in filter_name or "HP" in filter_name:
            sliders_column.controls = [
                d_label, d_slider, d_text
            ]
            sliders_column.visible = True
        else:
            sliders_column.visible = False

        page.update()
        update_filter()

    # Función para aplicar el filtro seleccionado
    def update_filter():
        if not selected_image_path:
            return
        image = cv.imread(selected_image_path, cv.IMREAD_GRAYSCALE)
        filter_name = filter_dropdown.value
        d_value = int(d_slider.value)
        n_value = int(n_slider.value)

        # Aplicar filtro basado en selección
        if filter_name == "Butterworth LP":
            processed_image = apply_filter(image, create_butterworth_low_pass_filter, d=d_value, n=n_value)
        elif filter_name == "Butterworth HP":
            processed_image = apply_filter(image, butterworth_high_pass_filter, d=d_value, n=n_value)
        elif filter_name == "Gaussian LP":
            processed_image = apply_filter(image, create_gaussian_low_pass_filter, d=d_value)
        elif filter_name == "Gaussian HP":
            processed_image = apply_filter(image, gaussian_high_pass_filter, d=d_value)
        elif filter_name == "Ideal LP":
            processed_image = apply_filter(image, create_ideal_low_pass_filter, d=d_value)
        elif filter_name == "Ideal HP":
            processed_image = apply_filter(image, ideal_high_pass_filter, d=d_value)
        else:
            return
        
        # Guardar resultado temporalmente
        processed_temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv.imwrite(processed_temp.name, processed_image)
        processed_image_display.src = processed_temp.name
        page.update()

    # Función de selección de archivo
    def on_file_selected(e):
        nonlocal selected_image_path
        if e.files:
            selected_image_path = e.files[0].path
        
            # Leer la imagen en escala de grises
            gray_image = cv.imread(selected_image_path, cv.IMREAD_GRAYSCALE)
        
            # Guardar la imagen en escala de grises temporalmente
            gray_temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv.imwrite(gray_temp.name, gray_image)
        
            # Mostrar la imagen en blanco y negro
            image_display.src = gray_temp.name
            processed_image_display.src = ""  # Limpiar la imagen procesada
        
            page.update()

    # Dropdown para seleccionar el filtro
    filter_dropdown = ft.Dropdown(
        label="Selecciona un filtro",
        options=[
            ft.dropdown.Option("Butterworth LP"),
            ft.dropdown.Option("Butterworth HP"),
            ft.dropdown.Option("Gaussian LP"),
            ft.dropdown.Option("Gaussian HP"),
            ft.dropdown.Option("Ideal LP"),
            ft.dropdown.Option("Ideal HP"),
        ],
        on_change=lambda e: toggle_sliders(e.control.value),
    )

    # Configurar FilePicker
    file_picker = ft.FilePicker(on_result=on_file_selected)
    page.overlay.append(file_picker)

    # Botón para seleccionar la imagen
    select_button = ft.ElevatedButton(
        "Seleccionar Imagen",
        on_click=lambda _: file_picker.pick_files(allowed_extensions=["png", "jpg", "jpeg"])
    )

    # Agregar componentes a la interfaz
    page.add(
        ft.Column(
            controls=[
                select_button,
                filter_dropdown,
                sliders_column,
                ft.Row(
                    controls=[image_display, processed_image_display],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )

# Ejecutar la aplicación
ft.app(target=main)
