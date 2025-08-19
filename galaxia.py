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
    """Calcula Σ(r)=M/Área en anillos concéntricos (plano XY).
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

    # Comprobaciones previas 
    if archivo_cargado is None:
        tk.messagebox.showwarning("Advertencia", "Primero carga un archivo"); return
    for w in frame_resultados.winfo_children():
        w.destroy()
    if np.sum(masas_completas) == 0:
        tk.messagebox.showwarning("Advertencia", "La suma de masas es cero"); return
    if 2 not in indices_componentes or len(indices_componentes[2]) == 0:
        tk.Label(frame_resultados, text="No se encontró componente de disco", fg="red").pack(); return

    #Datos del disco
    idx = indices_componentes[2]
    pos_d = posiciones_completas[idx]
    vel_d = velocidades_completas[idx]
    m_d   = masas_completas[idx]

    cm_d  = np.average(pos_d, axis=0, weights=m_d)
    pos_c = pos_d - cm_d
    L_d   = np.sum(np.cross(pos_c, vel_d) * m_d[:,None], axis=0)

    graficar_proyecciones_con_flecha(frame_resultados, pos_c, L_d, "Disco antes de rotar")
    graficar_3D_con_momento(frame_resultados, pos_c, L_d, "Disco 3D antes de rotar")

    # 1ª rotación: L -> Z 
    R1 = matriz_rotacion_secuencial(L_d)
    pos_r1 = pos_c @ R1.T
    vel_r1 = vel_d @ R1.T
    L_r1   = np.sum(np.cross(pos_r1, vel_r1) * m_d[:,None], axis=0)

    graficar_proyecciones_con_flecha(frame_resultados, pos_r1, L_r1, "Disco tras 1ª rotación (L→Z)")
    graficar_3D_con_momento(frame_resultados, pos_r1, L_r1, "Disco 3D tras 1ª rotación")

    #  2ª rotación: eliminar beta 
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
    import traceback
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    try:
        # Validaciones iniciales
        if archivo_cargado is None:
            tk.messagebox.showwarning("Advertencia", "Primero carga un archivo.")
            return
        if rotacion is None:
            tk.messagebox.showwarning("Advertencia", "Primero pulsa 'Mostrar orientación del disco'.")
            return
        if indices_componentes is None:
            tk.messagebox.showwarning("Advertencia", "No hay componentes cargados.")
            return

        # Limpiar frame
        for w in frame_resultados.winfo_children():
            w.destroy()
        tk.Label(frame_resultados, text="Calculando perfiles...", fg="blue").pack()

        # Datos y CM del disco
        dr = parametros_analisis['dr']
        radio_max = parametros_analisis['radio_max']
        idx_disco = indices_componentes.get(2, [])
        if len(idx_disco) == 0:
            tk.messagebox.showwarning("Advertencia", "No hay disco para referencia.")
            return
        cm_disco = np.average(posiciones_completas[idx_disco], axis=0, weights=masas_completas[idx_disco])

        # Diccionarios para almacenar resultados
        dfs_volumetricos = {}
        dfs_superficiales = {}
        dfs_lineales = {}
        dfs_velocidad_superficial = {}
        dfs_potencial_grav = {}

        G = 4.30091e-6  # kpc·(km/s)^2 / Msun

        # Procesar componentes
        componentes = [('Disco', 2), ('Bulbo', 3), ('Materia Oscura', 1)]
        for nombre, tipo in componentes:
            clave_sel = nombre.lower().replace(' ', '_')
            if not componentes_seleccionados.get(clave_sel, True):
                continue
            idx = indices_componentes.get(tipo, [])
            if len(idx) == 0:
                continue

            pos = posiciones_completas[idx]
            vel = velocidades_completas[idx]
            masas = masas_completas[idx]

            # Rotación respecto al CM del disco
            pos_r = (pos - cm_disco) @ rotacion.T
            vel_r = vel @ rotacion.T

            df = pd.DataFrame(pos_r, columns=['x_f','y_f','z_f'])
            df['vx_f'], df['vy_f'], df['vz_f'] = vel_r[:,0], vel_r[:,1], vel_r[:,2]
            df['masa'] = masas
            df['r_xy'] = np.sqrt(df['x_f']**2 + df['y_f']**2)
            df = df[df['r_xy'] < radio_max]

            # Perfil volumétrico
            bins_r = np.arange(0, df['r_xy'].max() + dr, dr)
            df['arandela'] = pd.cut(df['r_xy'], bins=bins_r, include_lowest=True, right=False, labels=False)
            perfil_vol = df.groupby('arandela', observed=True).sum().reset_index()
            radio_medio = 0.5 * (bins_r[:-1] + bins_r[1:])
            perfil_vol['radio_medio'] = radio_medio
            perfil_vol['radio_medio_log10'] = np.log10(perfil_vol['radio_medio'])
            perfil_vol['densidad_log10'] = np.log10(perfil_vol['masa'] / (4/3 * np.pi * (bins_r[1:]**3 - bins_r[:-1]**3)))
            perfil_vol.dropna(inplace=True)
            dfs_volumetricos[clave_sel] = perfil_vol
            perfil_vol.to_csv(f'perfil_volumetrico_{clave_sel}.csv', index=False)

            # Perfil superficial
            area_anillos = np.pi * (bins_r[1:]**2 - bins_r[:-1]**2)
            perfil_sup = perfil_vol.copy()
            perfil_sup['densidad_log10'] = np.log10(perfil_vol['masa'] / area_anillos)
            dfs_superficiales[clave_sel] = perfil_sup
            perfil_sup.to_csv(f'perfil_superficial_{clave_sel}.csv', index=False)

            # Perfil lineal solo disco
            if tipo == 2:
                bins_z = np.arange(df['z_f'].min(), df['z_f'].max() + dr, dr)
                mass_z, _ = np.histogram(df['z_f'], bins=bins_z, weights=df['masa'])
                centro_z = 0.5 * (bins_z[1:] + bins_z[:-1])
                perfil_lineal = pd.DataFrame({'z_medio': centro_z, 'densidad': mass_z})
                perfil_lineal['densidad_log10'] = np.log10(perfil_lineal['densidad'])
                dfs_lineales[clave_sel] = perfil_lineal
                perfil_lineal.to_csv(f'perfil_lineal_{clave_sel}.csv', index=False)

            # Velocidad superficial
            v_circ = np.sqrt(df['vx_f']**2 + df['vy_f']**2)
            perfil_vel = df.groupby('arandela', observed=True).apply(lambda g: np.average(v_circ[g.index], weights=g['masa'])).reset_index()
            perfil_vel.columns = ['arandela','v_circ']
            perfil_vel['radio_medio'] = radio_medio
            dfs_velocidad_superficial[clave_sel] = perfil_vel
            perfil_vel.to_csv(f'perfil_velocidad_superficial_{clave_sel}.csv', index=False)

            # Potencial gravitacional
            M_en_r = np.cumsum(perfil_vol['masa'].values)
            Phi = -G * M_en_r / (radio_medio + 1e-3)
            perfil_phi = pd.DataFrame({'radio_medio': radio_medio, 'Phi': Phi})
            dfs_potencial_grav[clave_sel] = perfil_phi
            perfil_phi.to_csv(f'potencial_gravitacional_{clave_sel}.csv', index=False)

        # Graficar todos los perfiles
        fig, ax = plt.subplots(3, 2, figsize=(12, 10))
        ax = ax.flatten()
        colores = {'disco':'blue', 'bulbo':'green', 'materia_oscura':'purple'}

        for clave_sel in dfs_volumetricos:
            sns.lineplot(dfs_volumetricos[clave_sel], x='radio_medio_log10', y='densidad_log10', 
                         color=colores.get(clave_sel,'black'), ax=ax[0], label=clave_sel)
        ax[0].set_title('Perfil volumétrico'); ax[0].set_xlabel('log10(R [kpc])'); ax[0].set_ylabel('log10(ρ)'); ax[0].legend()

        for clave_sel in dfs_superficiales:
            sns.lineplot(dfs_superficiales[clave_sel], x='radio_medio', y='densidad_log10', 
                         color=colores.get(clave_sel,'black'), ax=ax[1], label=clave_sel)
        ax[1].set_title('Perfil superficial'); ax[1].set_xlabel('R [kpc]'); ax[1].set_ylabel('log10(Σ)'); ax[1].legend()

        for clave_sel in dfs_lineales:
            sns.lineplot(dfs_lineales[clave_sel], x='z_medio', y='densidad_log10', 
                         color=colores.get(clave_sel,'black'), ax=ax[2], label=clave_sel)
        ax[2].set_title('Perfil lineal'); ax[2].set_xlabel('z [kpc]'); ax[2].set_ylabel('log10(ρ)'); ax[2].legend()

        for clave_sel in dfs_velocidad_superficial:
            sns.lineplot(dfs_velocidad_superficial[clave_sel], x='radio_medio', y='v_circ', 
                         color=colores.get(clave_sel,'black'), ax=ax[3], label=clave_sel)
        ax[3].set_title('Velocidad superficial'); ax[3].set_xlabel('R [kpc]'); ax[3].set_ylabel('v_circ [km/s]'); ax[3].legend()

        for clave_sel in dfs_potencial_grav:
            sns.lineplot(dfs_potencial_grav[clave_sel], x='radio_medio', y='Phi', 
                         color=colores.get(clave_sel,'black'), ax=ax[4], label=clave_sel)
        ax[4].set_title('Potencial gravitacional'); ax[4].set_xlabel('R [kpc]'); ax[4].set_ylabel('Φ [km²/s²]'); ax[4].legend()

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame_resultados)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        tk.Label(frame_resultados, text='Perfiles calculados y exportados (CSV) en la carpeta de trabajo.', fg='green').pack()

    except Exception as e:
        tk.messagebox.showerror("Error", f"{e}\n\n{traceback.format_exc()}")


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
tk.Button(frame_botones, text="Mostrar orientación del disco y rotar", command=mostrar_proyecciones).grid(row=2, column=0, columnspan=2, pady=5)
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