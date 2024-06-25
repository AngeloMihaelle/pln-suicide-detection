import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# importar el uso del modelo
from load_and_predict import classify_single_text

class MyWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Text sorter - suicide")
        self.root.geometry("800x500")

        # Estilo de la fuente
        self.normal_font = ("Arial", 12)
        self.title_font = ("Arial", 20)

        # Frame principal
        self.container = tk.Frame(root, bg="#f0f0f0")
        self.container.pack(fill="both", expand=True)

        # Frame de la barra lateral
        self.sidebar = tk.Frame(self.container, bg="#343A47", width=240)
        self.sidebar.pack(side="left", fill="y")

        # Frame del contenido
        self.contenido = tk.Frame(self.container, bg="#F5E2DC")
        self.contenido.pack(side="right", fill="both", expand=True)

        # Mostrar imagen en el sidebar
        self.show_image_in_sidebar()

        # Crear secciones de la barra lateral
        self.create_sections()

        # Configurar redimensionamiento responsivo
        self.root.bind("<Configure>", lambda event: self.resize_event())
        
        # Mostrar la sección "Classify" por defecto al iniciar
        self.show_classify_section()

    def show_image_in_sidebar(self):
        # Ruta de la imagen y tamaño máximo
        image_path = "./PLN_Proyecto/pato_Angello.jpg"  # Reemplaza con la ruta de tu imagen
        image = Image.open(image_path)
        image = image.resize((150, 150))  # Redimensionar la imagen a 10x10 píxeles
        image_photo = ImageTk.PhotoImage(image)

        # Mostrar la imagen con un margen de 10 píxeles
        label_image = tk.Label(self.sidebar, image=image_photo, bg="#343A47")
        label_image.image = image_photo  # Guardar la referencia para evitar que Python la elimine
        label_image.pack(side="top", padx=10, pady=(10, 10))  # Añadir margen de 10 píxeles alrededor de la imagen

    def create_sections(self):
        # Contenedor para "Classify"
        section_frame_1_container = tk.Frame(self.sidebar, bg="#EEB4B0", cursor="hand2")
        section_frame_1_container.pack(side="top", fill="x", padx=0, pady=0)
        
        # Label "Classify"
        classify_label = tk.Label(section_frame_1_container, text="Classify", font=self.normal_font, bg="#EEB4B0", fg="white")
        classify_label.pack(fill="both", expand=True, padx=10, pady=10)  # Añadimos padding al label
        classify_label.bind("<Button-1>", lambda event: self.show_classify_section())

        # Contenedor para "Exit"
        section_frame_2_container = tk.Frame(self.sidebar, bg="#343A47", cursor="hand2")
        section_frame_2_container.pack(side="top", fill="x", padx=0, pady=0)
        
        # Label "Exit"
        exit_label = tk.Label(section_frame_2_container, text="Exit", font=self.normal_font, bg="#343A47", fg="white")
        exit_label.pack(fill="both", expand=True, padx=10, pady=10)  # Añadimos padding al label
        exit_label.bind("<Button-1>", lambda event: self.exit_application())

    def show_classify_section(self):
        # Limpiar el contenido actual
        self.clear_content()

        # Mostrar sección de Classify
        classify_title = tk.Label(self.contenido, text="Classify", font=self.title_font, bg="#F5E2DC")
        classify_title.pack(pady=5)

        text_label = tk.Label(self.contenido, text="Insert or write a text to classify:", font=self.normal_font, bg="#F5E2DC")
        text_label.pack(pady=5)

        self.text_area = tk.Text(self.contenido, height=7, wrap="word")
        self.text_area.pack(pady=5, padx=5, fill="both", expand=True)

        classify_button = tk.Button(self.contenido, text="Classify", font=self.normal_font, bg="#343A47", fg="white", command=self.classify_text)
        classify_button.pack(pady=5)

        classification_label = tk.Label(self.contenido, text="Classification:", font=self.normal_font, bg="#F5E2DC")
        classification_label.pack(pady=5)

        self.result_area = tk.Text(self.contenido, height=4, wrap="word")
        self.result_area.pack(pady=5, padx=5, fill="both", expand=True)

    def classify_text(self):
        # Obtiene el texto del text area
        text_to_classify = self.text_area.get("1.0", "end-1c")
        # Usamos el modelo para predecir
        OUTPUT_DIR = './trained_model'  # Path to the directory containing the fine-tuned model
        predicted_label = classify_single_text(text_to_classify, model_path=OUTPUT_DIR)
        print(f"Predicted label from model: {predicted_label}")
        classification_result = self.classify_logic(predicted_label)
        
        # Versión final para mostrar
        final = ""
        if len(text_to_classify) <= 100:
            # Si el texto es menor o igual a 100 caracteres, imprímelo completo con un punto final.
            print(text_to_classify + '.')
            final = text_to_classify + '.'
        else:
            # Si el texto es mayor a 100 caracteres, recórtalo hasta el 100º caracter y agrega '...'
            print(text_to_classify[:100] + '...')
            final = text_to_classify[:100] + '...'


        # Formatea el texto a mostrar
        formatted_text = (
            f"Texto Clasificado:\n"
            f"{final}\n\n"
            f"Clasificación:\n"
            f"{classification_result}"
        )

        # Mostrar el texto formateado en el result_area
        self.result_area.delete("1.0", "end")
        self.result_area.insert("1.0", formatted_text)

    def classify_logic(self, text):
        """
        Dar formarto a la respuesta de la clasificación
        """
        
        if text.lower() == "suicida":
            return "El texto proporcionado muestra la presencia de tendencias suicidas."
        else:
            return "El texto proporcionado no muestra la presencia de tendencias suicidas."

    def clear_content(self):
        # Limpiar todos los widgets del contenido
        for widget in self.contenido.winfo_children():
            widget.pack_forget()

    def exit_application(self):
        # Cerrar la aplicación
        if messagebox.askokcancel("Salir", "¿Está seguro que desea salir?"):
            self.root.destroy()

    def resize_event(self, event=None):
        # Actualizar el tamaño del sidebar y del contenido al cambiar el tamaño de la ventana
        sidebar_width = int(self.root.winfo_width() * 0.3)
        contenido_width = self.root.winfo_width() - sidebar_width
        self.sidebar.config(width=sidebar_width)
        self.contenido.config(width=contenido_width)

if __name__ == "__main__":
    root = tk.Tk()
    app = MyWindow(root)
    root.mainloop()
