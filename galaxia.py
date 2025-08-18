import numpy as np
import struct
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

# Configuración de estilo de matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5
})

# Variables globales
archivo_cargado = None
posiciones_completas = None
velocidades_completas = None
masas_completas = None
indices_componentes = None
rotacion = None
parametros_analisis = {'dr': 0.2, 'radio_max': 30.0}
componentes_seleccionados = {'disco': True, 'bulbo': True, 'materia_oscura': True}

def calcular_perfil_densidad_superficial(radios_xy, masas, delta_r=0.2):
    """Calcula Σ(r)=M/Área en anillos concéntricos (plano XY).
       Devuelve centros de anillo y Σ.
    """
    radios_xy = np.asarray(radios_xy)
    masas     = np.asarray(masas)
    r_max     = np.max(radios_xy)
    bordes    = np.arange(0.0, r_max + delta_r, delta_r)
    centros   = 0.5 * (bordes[1:] + bordes[:-1])

    masa_bin  = np.zeros_like(centros)
    idx_bin   = np.digitize(radios_xy, bordes) - 1
    for i in range(len(centros)):
        masa_bin[i] = masas[idx_bin == i].sum()

    area_anillos = np.pi * (bordes[1:]**2 - bordes[:-1]**2)
    sigma        = np.divide(masa_bin, area_anillos, out=np.zeros_like(masa_bin), where=area_anillos>0)
    return centros, sigma

def graficar_3D_con_momento(frame, posiciones, momento, titulo):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(posiciones[:, 0], posiciones[:, 1], posiciones[:, 2], s=0.1, alpha=0.3)
    ax.quiver(0, 0, 0, *momento, color='red', label='Momento Angular', linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, np.linalg.norm(momento), color='blue', linestyle='dashed', label='Eje Z', linewidth=2)
    ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_zlim(-100, 100)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title(titulo); ax.legend()
    FigureCanvasTkAgg(fig, master=frame).get_tk_widget().pack()

# Funciones para leer archivo GADGET-2 y procesar datos
def leer_snapshot(nombre_archivo):
    with open(nombre_archivo, "rb") as archivo:
        archivo.read(4)
        datos_cabecera = archivo.read(256)
        archivo.read(4)

        num_particulas = struct.unpack("6I", datos_cabecera[0:24])
        masas = struct.unpack("6d", datos_cabecera[24:72])

        total_particulas = sum(num_particulas)

        archivo.read(4)
        posiciones = np.fromfile(archivo, dtype=np.float32, count=total_particulas * 3).reshape((total_particulas, 3))
        archivo.read(4)

        archivo.read(4)
        velocidades = np.fromfile(archivo, dtype=np.float32, count=total_particulas * 3).reshape((total_particulas, 3))
        archivo.read(4)

        archivo.read(4)
        ids = np.fromfile(archivo, dtype=np.uint32, count=total_particulas)
        archivo.read(4)

        return posiciones, velocidades, ids, num_particulas, masas

def indices_por_tipo(num_particulas):
    indices = {}
    inicio = 0
    for tipo in range(6):
        cantidad = num_particulas[tipo]
        indices[tipo] = np.arange(inicio, inicio + cantidad)
        inicio += cantidad
    return indices

def calcular_centro_masa(posiciones, masas):
    if len(posiciones) == 0 or np.sum(masas) == 0:
        return np.zeros(3)
    return np.average(posiciones, axis=0, weights=masas)

def calcular_momento_angular(posiciones, velocidades, masas):
    return np.sum(np.cross(posiciones, velocidades) * masas[:, None], axis=0)

def calcular_angulos(vector):
    modulo = np.linalg.norm(vector)
    lz = vector[2]
    alpha = np.arccos(lz / modulo) if modulo != 0 else 0
    beta = np.arctan2(vector[1], vector[0])
    return alpha, beta

def matriz_rotacion_secuencial(L):
    L = L / np.linalg.norm(L)
    lx, ly, lz = L

    # Rotación alrededor del eje Y para alinear en el plano XZ
    theta_y = np.arctan2(lx, lz)
    Ry = np.array([
        [np.cos(theta_y), 0, -np.sin(theta_y)],
        [0, 1, 0],
        [np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    # Aplicar Ry a L
    L_1 = Ry @ L

    # Rotación alrededor del eje X para alinear completamente con Z
    theta_x = np.arctan2(L_1[1], L_1[2])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    return Rx @ Ry

def graficar_proyecciones_con_flecha(frame, posiciones, momento, titulo):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    planos = [(0, 1), (1, 2), (0, 2)]
    nombres = ["Plano XY", "Plano YZ", "Plano XZ"]

    for i, (x_idx, y_idx) in enumerate(planos):
        axs[i].scatter(posiciones[:, x_idx], posiciones[:, y_idx], s=0.1)
        axs[i].set_title(nombres[i])
        axs[i].set_xlim(-100, 100)
        axs[i].set_ylim(-100, 100)
        axs[i].arrow(0, 0, momento[x_idx], momento[y_idx], color='red', width=0.5, head_width=2)

    fig.suptitle(titulo)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Funciones de interfaz
def cargar_archivo():
    global archivo_cargado, posiciones_completas, velocidades_completas, masas_completas, indices_componentes

    archivo = filedialog.askopenfilename()
    if not archivo:
        return

    try:
        posiciones, velocidades, ids, num_particulas, masas_tipos = leer_snapshot(archivo)
        indices_componentes = indices_por_tipo(num_particulas)

        posiciones_completas = posiciones
        velocidades_completas = velocidades

        masas_completas = np.zeros(len(posiciones))
        for tipo in range(6):
            if tipo in indices_componentes and len(indices_componentes[tipo]) > 0:
                if masas_tipos[tipo] != 0:
                    masas_completas[indices_componentes[tipo]] = masas_tipos[tipo]
                else:
                    masas_completas[indices_componentes[tipo]] = 1.0

        archivo_cargado = archivo
    except Exception as e:
        tk.messagebox.showerror("Error", f"No se pudo cargar el archivo: {str(e)}")

def mostrar_proyecciones():
    """Muestra proyecciones, aplica doble rotación (L→Z y β→0)
    y calcula el perfil Σ(r) filtrando radios > 30 kpc."""
    global rotacion

    # ------ Comprobaciones previas ------
    if archivo_cargado is None:
        tk.messagebox.showwarning("Advertencia", "Primero carga un archivo"); return
    for w in frame_resultados.winfo_children():
        w.destroy()
    if np.sum(masas_completas) == 0:
        tk.messagebox.showwarning("Advertencia", "La suma de masas es cero"); return
    if 2 not in indices_componentes or len(indices_componentes[2]) == 0:
        tk.Label(frame_resultados, text="No se encontró componente de disco", fg="red").pack(); return

    # ------ Datos del disco ------
    idx = indices_componentes[2]
    pos_d = posiciones_completas[idx]
    vel_d = velocidades_completas[idx]
    m_d   = masas_completas[idx]

    cm_d  = np.average(pos_d, axis=0, weights=m_d)
    pos_c = pos_d - cm_d
    L_d   = np.sum(np.cross(pos_c, vel_d) * m_d[:,None], axis=0)

    graficar_proyecciones_con_flecha(frame_resultados, pos_c, L_d, "Disco antes de rotar")
    graficar_3D_con_momento(frame_resultados, pos_c, L_d, "Disco 3D antes de rotar")

    # ------ 1ª rotación: L -> Z ------
    R1 = matriz_rotacion_secuencial(L_d)
    pos_r1 = pos_c @ R1.T
    vel_r1 = vel_d @ R1.T
    L_r1   = np.sum(np.cross(pos_r1, vel_r1) * m_d[:,None], axis=0)

    graficar_proyecciones_con_flecha(frame_resultados, pos_r1, L_r1, "Disco tras 1ª rotación (L→Z)")
    graficar_3D_con_momento(frame_resultados, pos_r1, L_r1, "Disco 3D tras 1ª rotación")

    # ------ 2ª rotación: eliminar beta ------
    _, beta_after = calcular_angulos(L_r1)
    Rz = np.array([[ np.cos(-beta_after), -np.sin(-beta_after), 0],
                   [ np.sin(-beta_after),  np.cos(-beta_after), 0],
                   [ 0,                   0,                    1]])
    pos_r = pos_r1 @ Rz.T
    vel_r = vel_r1 @ Rz.T
    L_r   = np.sum(np.cross(pos_r, vel_r) * m_d[:,None], axis=0)
    rotacion = Rz @ R1

    graficar_proyecciones_con_flecha(frame_resultados, pos_r, L_r, "Disco tras 2ª rotación (β→0)")
    graficar_3D_con_momento(frame_resultados, pos_r, L_r, "Disco 3D final")
    alpha_f, beta_f = calcular_angulos(L_r)
    tk.Label(frame_resultados, text=f"α final = {np.degrees(alpha_f):.3f}°, β final = {np.degrees(beta_f):.3f}°").pack()
    
def mostrar_resultados_y_exportar():
    idx_disco = indices_componentes[2]
    cm_d = np.average(posiciones_completas[idx_disco], axis=0, weights=masas_completas[idx_disco])

    if archivo_cargado is None or rotacion is None:
        return

    for w in frame_resultados.winfo_children():
        w.destroy()

    dr = parametros_analisis['dr']
    radio_max = parametros_analisis['radio_max']

    # Procesar cada componente
    for nombre, tipo in [('Disco', 2), ('Bulbo', 3), ('Materia Oscura', 1)]:
        if not componentes_seleccionados.get(nombre.lower().replace(' ', '_')):
            continue
        if tipo not in indices_componentes:
            continue

        idx = indices_componentes[tipo]
        pos = posiciones_completas[idx]
        masas = masas_completas[idx]
        cm_disco = np.average(posiciones_completas[indices_componentes[2]], axis=0, weights=masas_completas[indices_componentes[2]])
        pos_r = (pos - cm_disco) @ rotacion.T

        df = pd.DataFrame(pos_r, columns=['x_f', 'y_f', 'z_f'])
        df['masa'] = masas
        df['r_xy'] = np.sqrt(df['x_f']**2 + df['y_f']**2)
        df = df[df['r_xy'] < radio_max]

        # Perfil ρ(z)
        bins_z = np.arange(-30, 30 + 0.2, 0.2)
        df['bin_z'] = pd.cut(df['z_f'], bins=bins_z)
        perfil_z = df.groupby('bin_z')['masa'].sum().reset_index()
        centro_z = 0.5 * (bins_z[:-1] + bins_z[1:])
        area_proj = np.pi * radio_max**2
        perfil_z['densidad_z'] = perfil_z['masa'] / (0.2 * area_proj)
        perfil_z['z_centro'] = centro_z
        perfil_z = perfil_z[perfil_z['densidad_z'] > 0]
        perfil_z.to_csv(f'perfil_z_{nombre.lower().replace(" ", "_")}.csv', index=False)

        fig_z, ax_z = plt.subplots(figsize=(6, 4))
        ax_z.plot(perfil_z['z_centro'], np.log10(perfil_z['densidad_z']), label=nombre)
        ax_z.set_xlabel('z (kpc)')
        ax_z.set_ylabel(r'$\log_{10}(\rho\ [M_\odot / kpc^3])$')
        ax_z.set_title(f'Perfil de densidad vertical {nombre}')
        ax_b.set_xlim(-3, 3)
        ax_b.set_ylim(-2, 1)
        ax_b.autoscale(False)
        ax_z.legend()
        FigureCanvasTkAgg(fig_z, master=frame_resultados).get_tk_widget().pack()

        # Perfil Σ(r)
        bordes = np.arange(0, df['r_xy'].max() + dr, dr)
        df['arandela'] = pd.cut(df['r_xy'], bins=bordes)
        perfil_r = df.groupby('arandela')['masa'].sum().reset_index()
        radio_medio = 0.5 * (bordes[:-1] + bordes[1:])
        area = np.pi * (bordes[1:]**2 - bordes[:-1]**2)
        perfil_r['densidad'] = perfil_r['masa'] / area
        perfil_r['radio_medio'] = radio_medio
        perfil_r = perfil_r[perfil_r['densidad'] > 0]
        perfil_r.to_csv(f'perfil_{nombre.lower().replace(" ", "_")}.csv', index=False)

        fig_r, ax_r = plt.subplots(figsize=(6, 4))
        ax_r.plot(perfil_r['radio_medio'], perfil_r['densidad'])
        ax_r.set_xlabel(r'$\log_{10}(r\ [kpc])$')
        ax_r.set_ylabel(r'(\Sigma\ [M_\odot / kpc^2])$')
        ax_r.set_title(f'Perfil log-log {nombre}')
        ax_b.set_xlim(0, 25)
        ax_b.set_ylim(-5, -1)
        ax_b.autoscale(False)
        ax_r.legend()
        FigureCanvasTkAgg(fig_r, master=frame_resultados).get_tk_widget().pack()

    # ----------- PERFIL bulbo -----------
    perfil_b = None
    if componentes_seleccionados['bulbo'] and 3 in indices_componentes:
        idx_b = indices_componentes[3]
        pos_b = (posiciones_completas[idx_b] - cm_d) @ rotacion.T
        m_b = masas_completas[idx_b]

        df_bulbo = pd.DataFrame(pos_b, columns=['x_f', 'y_f', 'z_f'])
        df_bulbo['masa'] = m_b
        df_bulbo['r_xy'] = np.sqrt(df_bulbo.x_f**2 + df_bulbo.y_f**2)
        df_bulbo = df_bulbo[df_bulbo['r_xy'] < parametros_analisis['radio_max']]

        bordes_b = np.arange(0, df_bulbo.r_xy.max() + dr, dr)
        suma_rad_b = bordes_b[1:] + bordes_b[:-1]
        area_b = np.pi * dr * suma_rad_b

        df_bulbo['arandela'] = pd.cut(df_bulbo.r_xy, bins=bordes_b, labels=np.arange(len(bordes_b)-1))
        perfil_b = df_bulbo.groupby('arandela')['masa'].sum().to_frame()
        perfil_b['area'] = area_b
        perfil_b['densidad'] = perfil_b['masa'] / perfil_b['area']
        perfil_b['radio_medio'] = suma_rad_b / 2
        perfil_b = perfil_b[perfil_b['densidad'] > 0]
        perfil_b.to_csv('perfil_bulbo.csv', index=False)

        fig_b, ax_b = plt.subplots(figsize=(6, 4))
        ax_b.plot(np.log10(perfil_b['radio_medio']), np.log10(perfil_b['densidad']), label='Bulbo', color='purple')
        ax_b.set_xlabel(r'$\log_{10}(r\ [kpc])$')
        ax_b.set_ylabel(r'$\log_{10}(\Sigma\ [M_\odot / kpc^2])$')
        ax_b.set_title('Perfil log-log del bulbo')
        ax_b.set_xlim(-2, 1)
        ax_b.set_ylim(-5, 2)
        ax_b.autoscale(False)
        ax_b.legend()
        FigureCanvasTkAgg(fig_b, master=frame_resultados).get_tk_widget().pack()


        tk.Label(frame_resultados, text='Perfil Σ bulbo exportado en perfil_bulbo.csv', fg='blue').pack()
    else:
        tk.Label(frame_resultados, text="No se encontró componente de bulbo o no está seleccionado", fg="gray").pack()

    # ----------- PERFIL materia oscura -----------
    perfil_dm = None
    if componentes_seleccionados['materia_oscura'] and 1 in indices_componentes:
        idx_dm = indices_componentes[1]
        pos_dm = (posiciones_completas[idx_dm] - cm_d) @ rotacion.T
        m_dm = masas_completas[idx_dm]

        df_dm = pd.DataFrame(pos_dm, columns=['x_f', 'y_f', 'z_f'])
        df_dm['masa'] = m_dm
        df_dm['r_xy'] = np.sqrt(df_dm.x_f**2 + df_dm.y_f**2)
        df_dm = df_dm[df_dm['r_xy'] < parametros_analisis['radio_max']]

        bordes_dm = np.arange(0, df_dm.r_xy.max() + dr, dr)
        suma_rad_dm = bordes_dm[1:] + bordes_dm[:-1]
        area_dm = np.pi * dr * suma_rad_dm

        df_dm['arandela'] = pd.cut(df_dm.r_xy, bins=bordes_dm, labels=np.arange(len(bordes_dm)-1))
        perfil_dm = df_dm.groupby('arandela')['masa'].sum().to_frame()
        perfil_dm['area'] = area_dm
        perfil_dm['densidad'] = perfil_dm['masa'] / perfil_dm['area']
        perfil_dm['radio_medio'] = suma_rad_dm / 2
        perfil_dm = perfil_dm[perfil_dm['densidad'] > 0]
        perfil_dm.to_csv('perfil_materia_oscura.csv', index=False)

        fig_dm, ax_dm = plt.subplots(figsize=(6, 4))
        ax_dm.plot(np.log10(perfil_dm['radio_medio']), np.log10(perfil_dm['densidad']), label='Materia Oscura', color='black')
        ax_dm.set_xlabel(r'$\log_{10}(r\ [kpc])$')
        ax_dm.set_ylabel(r'$\log_{10}(\Sigma\ [M_\odot / kpc^2])$')
        ax_dm.set_title('Perfil log-log de la materia oscura')
        ax_b.set_xlim(-5, 2)
        ax_b.set_ylim(-2, 1)
        ax_b.autoscale(False)
        ax_dm.legend()
        FigureCanvasTkAgg(fig_dm, master=frame_resultados).get_tk_widget().pack()


        tk.Label(frame_resultados, text='Perfil Σ materia oscura exportado en perfil_materia_oscura.csv', fg='blue').pack()
    else:
        tk.Label(frame_resultados, text="No se encontró componente de materia oscura o no está seleccionado", fg="gray").pack()

    # ----------- COMPARACIÓN FINAL -----------
    fig_cmp, ax_cmp = plt.subplots(figsize=(7, 5))
    if perfil_r is not None:
        ax_cmp.plot(np.log10(perfil_r['radio_medio']), np.log10(perfil_r['densidad']), label='Disco', color='blue')
    if perfil_b is not None:
        ax_cmp.plot(np.log10(perfil_b['radio_medio']), np.log10(perfil_b['densidad']), label='Bulbo', color='purple')
    if perfil_dm is not None:
        ax_cmp.plot(np.log10(perfil_dm['radio_medio']), np.log10(perfil_dm['densidad']), label='Materia Oscura', color='black')

    ax_cmp.set_xlabel(r'$\log_{10}(r\ [kpc])$')
    ax_cmp.set_ylabel(r'$\log_{10}(\Sigma\ [M_\odot / kpc^2])$')
    ax_cmp.set_title('Comparación de perfiles')
    ax_cmp.legend()
    FigureCanvasTkAgg(fig_cmp, master=frame_resultados).get_tk_widget().pack()




def seleccionar_componentes():
    ventana_componentes = tk.Toplevel(ventana)
    ventana_componentes.title("Seleccionar componentes")
    ventana_componentes.geometry("300x200")

    def actualizar():
        componentes_seleccionados['disco'] = var_disco.get()
        componentes_seleccionados['bulbo'] = var_bulbo.get()
        componentes_seleccionados['materia_oscura'] = var_dm.get()
        ventana_componentes.destroy()

    var_disco = tk.BooleanVar(value=componentes_seleccionados['disco'])
    var_bulbo = tk.BooleanVar(value=componentes_seleccionados['bulbo'])
    var_dm = tk.BooleanVar(value=componentes_seleccionados['materia_oscura'])

    tk.Checkbutton(ventana_componentes, text="Disco", variable=var_disco).pack(anchor='w')
    tk.Checkbutton(ventana_componentes, text="Bulbo", variable=var_bulbo).pack(anchor='w')
    tk.Checkbutton(ventana_componentes, text="Materia Oscura", variable=var_dm).pack(anchor='w')
    tk.Button(ventana_componentes, text="Guardar", command=actualizar).pack(pady=10)

def definir_parametros():
    ventana_param = tk.Toplevel(ventana)
    ventana_param.title("Parámetros de análisis")
    ventana_param.geometry("300x200")

    tk.Label(ventana_param, text="Ancho de anillo dr (kpc):").pack()
    entry_dr = tk.Entry(ventana_param)
    entry_dr.insert(0, str(parametros_analisis['dr']))
    entry_dr.pack()

    tk.Label(ventana_param, text="Radio máximo (kpc):").pack()
    entry_rmax = tk.Entry(ventana_param)
    entry_rmax.insert(0, str(parametros_analisis['radio_max']))
    entry_rmax.pack()

    def guardar():
        try:
            parametros_analisis['dr'] = float(entry_dr.get())
            parametros_analisis['radio_max'] = float(entry_rmax.get())
            ventana_param.destroy()
        except ValueError:
            tk.messagebox.showerror("Error", "Los parámetros deben ser numéricos")

    tk.Button(ventana_param, text="Guardar", command=guardar).pack(pady=10)

def mostrar_explicacion():
    ventana_exp = tk.Toplevel(ventana)
    ventana_exp.title("Explicación del algoritmo")
    ventana_exp.geometry("600x400")
    texto = """
1. Se selecciona el componente de disco para determinar el momento angular.
2. Se traslada al centro de masa del disco.
3. Se aplica una rotación secuencial para alinear el momento angular con el eje Z.
4. Se calcula un perfil vertical ρ(z) y superficial Σ(r) para cada componente seleccionado.
5. Los resultados se exportan y visualizan en escala log-log.
    """
    tk.Label(ventana_exp, text=texto, justify="left", wraplength=580).pack(padx=10, pady=10)

# Crear ventana principal
ventana = tk.Tk()
ventana.title("Análisis de Galaxias")
ventana.geometry("1300x900")

frame_botones = tk.Frame(ventana)
frame_botones.pack(pady=20)

tk.Label(frame_botones, text="Selecciona el tipo de archivo:").grid(row=0, column=0)
tipo_archivo_var = tk.StringVar(value="Binario")
tipo_archivo_menu = tk.OptionMenu(frame_botones, tipo_archivo_var, "Binario", "ASCII")
tipo_archivo_menu.grid(row=0, column=1)

tk.Button(frame_botones, text="Cargar archivo", command=cargar_archivo).grid(row=1, column=0, columnspan=2, pady=5)
tk.Button(frame_botones, text="Mostrar orientación del disco", command=mostrar_proyecciones).grid(row=2, column=0, columnspan=2, pady=5)
tk.Button(frame_botones, text="Seleccionar componentes galácticos", command=seleccionar_componentes).grid(row=3, column=0, columnspan=2, pady=5)
tk.Button(frame_botones, text="Parámetros de análisis", command=definir_parametros).grid(row=4, column=0, columnspan=2, pady=5)
tk.Button(frame_botones, text="Explicación visual del algoritmo", command=mostrar_explicacion).grid(row=5, column=0, columnspan=2, pady=5)
tk.Button(frame_botones, text="Calcular perfiles y exportar CSV", command=mostrar_resultados_y_exportar).grid(row=6, column=0, columnspan=2, pady=5)
tk.Button(frame_botones, text="Cerrar aplicación", command=ventana.destroy).grid(row=7, column=0, columnspan=2, pady=5)


frame_scroll = tk.Frame(ventana)
frame_scroll.pack(fill="both", expand=True)

canvas_scroll = tk.Canvas(frame_scroll)
scrollbar = tk.Scrollbar(frame_scroll, orient="vertical", command=canvas_scroll.yview)
scrollable_frame = tk.Frame(canvas_scroll)
scrollable_frame.bind("<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas_scroll.configure(yscrollcommand=scrollbar.set)

canvas_scroll.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

frame_resultados = scrollable_frame
ventana.mainloop()