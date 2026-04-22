import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
import mplcursors 

class Vista:
    def encabezado(self):
        print("\n╔══════════════════════════════════════════╗")
        print("║   PERCEPTRÓN 3D (MVC) – Clasificación    ║")
        print("╚══════════════════════════════════════════╝\n")

    def pedir_condiciones_iniciales(self):
        print("Ingresa las condiciones iniciales (Ej: eta=1, W1=1, W2=1, W3=1, W4=1):")
        eta = float(input("  eta (tasa de aprendizaje): "))
        w1 = float(input("  W1 (peso del sesgo X0=1): "))
        w2 = float(input("  W2 (peso de entrada A   ): "))
        w3 = float(input("  W3 (peso de entrada B   ): "))
        w4 = float(input("  W4 (peso de entrada C   ): "))
        return eta, [w1, w2, w3, w4]

    def mostrar_resultados(self, ha_convergido, iteracion, W):
        print("\n============================================")
        if ha_convergido:
            print(f"  Convergencia alcanzada en {iteracion} iteraciones")
        else:
            print(f"  SIN convergencia tras {iteracion} iteraciones")
        print("--------------------------------------------")
        print(f"  Pesos finales: W = [{W[0]:+.4f}, {W[1]:+.4f}, {W[2]:+.4f}, {W[3]:+.4f}]")
        print("============================================\n")

    def mostrar_historial(self, historial):
        print("\n============================================")
        print("  TRAZA DE ENTRENAMIENTO")
        print("============================================")
        for entrada in historial:
            it = entrada['iteracion']
            errores = entrada['errores']
            cambios = entrada['cambios']
            if cambios:
                print(f"\n  Iteración {it}  (errores: {errores})")
                print(f"  {'Patrón':<8} {'A':>2} {'B':>2} {'C':>2}  {'y':>2}  {'c':>2}  {'W resultante'}")
                print(f"  {'-'*60}")
                for ch in cambios:
                    W = ch['W']
                    Wstr = f"[{W[0]:+.2f}, {W[1]:+.2f}, {W[2]:+.2f}, {W[3]:+.2f}]"
                    print(f"  {ch['patron']:<8} {ch['A']:>2} {ch['B']:>2} {ch['C']:>2}  {ch['y']:>2}  {ch['c']:>2}  {Wstr}")
            else:
                print(f"\n  Iteración {it}  — sin errores => CONVERGIDO")
        print("============================================\n")

    def preguntar_repetir(self):
        respuesta = input("¿Lo quieres volver a intentar? (s/n): ").lower()
        while respuesta not in ['s', 'n']:
            respuesta = input("Respuesta no válida. Escribe s o n: ").lower()
        return respuesta == 's'

    def dibujar_grafica_3d(self, X, C_objetivo, W, eta, iteracion):
        fig = plt.figure('Perceptron 3D - Cubo', figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Perceptrón 3D | eta={eta:.4f} | Iter={iteracion}\nW=[{W[0]:.2f}, {W[1]:.2f}, {W[2]:.2f}, {W[3]:.2f}]")

        col_c1, col_c2 = '#3373CC', '#D93333'

        x0, y0, z0 = [], [], []
        x1, y1, z1 = [], [], []

        for i in range(len(X)):
            if C_objetivo[i] == 0:
                x0.append(X[i, 1]); y0.append(X[i, 2]); z0.append(X[i, 3])
            else:
                x1.append(X[i, 1]); y1.append(X[i, 2]); z1.append(X[i, 3])

        scat_c0 = ax.scatter(x0, y0, z0, color=col_c1, s=100, marker='o', edgecolors='black')
        scat_c1 = ax.scatter(x1, y1, z1, color=col_c2, s=100, marker='^', edgecolors='black')

        cursor = mplcursors.cursor([scat_c0, scat_c1], hover=True)
        
        @cursor.connect("add")
        def al_pasar_cursor(sel):
            idx = sel.index
            
            if sel.artist == scat_c0:
                a, b, c = x0[idx], y0[idx], z0[idx]
                clase = "1 (Círculos azules)"
            else:
                a, b, c = x1[idx], y1[idx], z1[idx]
                clase = "2 (Triángulos rojos)"
            
            texto = f"Entrada A: {a}\nEntrada B: {b}\nEntrada C: {c}\nClase: {clase}"
            sel.annotation.set_text(texto)
            
            sel.annotation.get_bbox_patch().set_facecolor('white')
            sel.annotation.get_bbox_patch().set_edgecolor('black')
            sel.annotation.get_bbox_patch().set_alpha(0.9)

        r = [0, 1]
        for inicio, fin in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(inicio - fin)) == 1:
                ax.plot3D(*zip(inicio, fin), color="gray", linewidth=1.2)

        if W[3] != 0:
            xx, yy = np.meshgrid([0, 1], [0, 1])
            zz = -(W[0] + W[1]*xx + W[2]*yy) / W[3]
            zz = np.clip(zz, 0, 1)
            ax.plot_surface(xx, yy, zz, alpha=0.5, color='lime', edgecolor='green')

        ax.set_xlabel('A (X_1)'); ax.set_ylabel('B (X_2)'); ax.set_zlabel('C (X_3)')
        ax.view_init(elev=25, azim=40)
        
        plt.show()