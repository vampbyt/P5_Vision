import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
import mplcursors 

class Vista:
    def pedir_condiciones_iniciales(self):
        print("\n╔════════════════════════════════════════╗")
        print("║   PERCEPTRÓN 3D                          ║")
        print("╚══════════════════════════════════════════╝\n")
        print("Por favor, ingresa las condiciones iniciales:")
        # Cambiamos los nombres para que sean idénticos a los que buscas
        r = float(input("  Valor de r (tasa de aprendizaje) : "))
        w1 = float(input("  Valor de W1 (peso de entrada A)  : "))
        w2 = float(input("  Valor de W2 (peso de entrada B)  : "))
        w3 = float(input("  Valor de W3 (peso de entrada C)  : "))
        w4 = float(input("  Valor de W4 (peso del Bias X0=1) : "))
        return r, [w1, w2, w3, w4]

    def imprimir_tabla_historial(self, historial, iteraciones, w_final):
        print("\n" + "="*80)
        print(f"{'TABLA DE APRENDIZAJE':^80}")
        print("="*80)
        print(f"{'N':<4} | {'Clase':<5} | {'Xn (A,B,C,1)':<14} | {'Wn_T':<16} | {'Net':<4} | {'Sanción':<8} | {'W_nuevo':<16}")
        print("-" * 80)
        
        for fila in historial:
            x_str = f"{int(fila['x'][0])} {int(fila['x'][1])} {int(fila['x'][2])} {int(fila['x'][3])}"
            w_ant_str = f"{int(fila['w_antes'][0]):2} {int(fila['w_antes'][1]):2} {int(fila['w_antes'][2]):2} {int(fila['w_antes'][3]):2}"
            w_nue_str = f"{int(fila['w_nuevo'][0]):2} {int(fila['w_nuevo'][1]):2} {int(fila['w_nuevo'][2]):2} {int(fila['w_nuevo'][3]):2}"
            
            if fila['net'] > 0: net_signo = ">0"
            elif fila['net'] < 0: net_signo = "<0"
            else: net_signo = "=0"
            
            print(f"{fila['paso']:<4} | {fila['clase']:<5} | {x_str:<14} | {w_ant_str:<16} | {net_signo:<4} | {fila['sancion']:<8} | {w_nue_str:<16}")
        
        print("="*80)
        print(f"▶ Resultado final en {iteraciones} iteraciones: W = [{w_final[0]:.0f}, {w_final[1]:.0f}, {w_final[2]:.0f}, {w_final[3]:.0f}]")
        print("="*80 + "\n")

    def dibujar_grafica_3d(self, X, C_objetivo, W):
        w_final_str = f"[{W[0]:.0f}, {W[1]:.0f}, {W[2]:.0f}, {W[3]:.0f}]"
        
        fig = plt.figure('Perceptron 3D - Cubo', figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Perceptrón - Cubo | Resultado Final\nW = {w_final_str}", fontsize=12, fontweight='bold')

        x0, y0, z0 = [], [], []
        x1, y1, z1 = [], [], []

        for i in range(len(X)):
            if C_objetivo[i] == 0:
                x0.append(X[i, 0]); y0.append(X[i, 1]); z0.append(X[i, 2])
            else:
                x1.append(X[i, 0]); y1.append(X[i, 1]); z1.append(X[i, 2])

        scat_c0 = ax.scatter(x0, y0, z0, color='#3373CC', s=100, marker='o', edgecolors='black', label='C1 (Objetivo < 0)')
        scat_c1 = ax.scatter(x1, y1, z1, color='#D93333', s=100, marker='^', edgecolors='black', label='C2 (Objetivo > 0)')

        cursor = mplcursors.cursor([scat_c0, scat_c1], hover=True)
        @cursor.connect("add")
        def al_pasar_cursor(sel):
            idx = sel.index
            if sel.artist == scat_c0:
                a, b, c = x0[idx], y0[idx], z0[idx]
                clase = "1 (C1 - Objetivo < 0)"
            else:
                a, b, c = x1[idx], y1[idx], z1[idx]
                clase = "2 (C2 - Objetivo > 0)"
            
            texto = f"Entrada A: {a}\nEntrada B: {b}\nEntrada C: {c}\nClase: {clase}"
            sel.annotation.set_text(texto)
            sel.annotation.get_bbox_patch().set_facecolor('white')
            sel.annotation.get_bbox_patch().set_edgecolor('black')
            sel.annotation.get_bbox_patch().set_alpha(0.9)

        r = [0, 1]
        for inicio, fin in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(inicio - fin)) == 1:
                ax.plot3D(*zip(inicio, fin), color="gray", linewidth=1.2)

        if W[2] != 0:
            xx, yy = np.meshgrid([0, 1], [0, 1])
            zz = -(W[3] + W[0]*xx + W[1]*yy) / W[2]
            zz = np.clip(zz, 0, 1)
            ax.plot_surface(xx, yy, zz, alpha=0.5, color='lime', edgecolor='green')

        ax.set_xlabel('A (X_1)'); ax.set_ylabel('B (X_2)'); ax.set_zlabel('C (X_3)')
        ax.view_init(elev=25, azim=40)
        plt.legend(loc='lower right')
        
        # Esto abre la ventana gráfica (y pausa la consola hasta que se cierre)
        plt.show()