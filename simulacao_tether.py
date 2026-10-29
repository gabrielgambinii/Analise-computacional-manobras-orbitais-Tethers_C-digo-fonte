"""
================================================================================
SIMULACAO DE MANOBRA ORBITAL COM TETHER ELETRODINAMICO
Camada de Guiamento (Problema Inverso)
Baseado em: Tragesser, S. G.; San, H. "Orbital Maneuvering with Electrodynamic
            Tethers." Journal of Guidance, Control, and Dynamics, v. 26, n. 5,
            p. 805-810, 2003.
================================================================================

Este script implementa apenas a CAMADA DE GUIAMENTO do metodo, isto e, a
resolucao algebrica do problema inverso: dada uma mudanca orbital desejada,
calcular a lei de corrente I(nu, omega) e o tempo de voo (TOF) necessarios.

A CAMADA DE SIMULACAO DINAMICA (integracao das equacoes de Gauss com a
corrente projetada) NAO esta implementada e fica reservada para trabalho
complementar futuro.

Estrutura:
  BLOCO 1 - Constantes fisicas do sistema
  BLOCO 2 - Funcoes matematicas do metodo (nucleo do Tragesser)
  BLOCO 3 - Funcoes de interface, entrada, saida e graficos
  BLOCO 4 - Programa principal (orquestracao)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra projecao 3D)
import os
import csv
import math
from datetime import datetime


# ============================================================================
# BLOCO 1 - CONSTANTES FISICAS DO SISTEMA
# ============================================================================
# Todas as constantes seguem os valores adotados por Tragesser e San (2003)
# para permitir comparacao direta com os exemplos numericos do artigo.
# Sistema de unidades: km, s, kg, A, T.
# ============================================================================

MU_TERRA = 3.986e5          # Parametro gravitacional da Terra [km^3/s^2]
R_TERRA = 6378.0            # Raio equatorial da Terra [km]
MU_M = 8.0e6                # Momento de dipolo magnetico da Terra [T * km^3]
                            # Valor explicito na eq. (3) do Tragesser (p. 805)
L_TETHER = 15.0             # Comprimento do tether [km] - Tragesser p. 808
M_SISTEMA = 1000.0          # Massa total do sistema tetherizado [kg] - p. 808
I_DISP = 5.0                # Corrente maxima disponivel a bordo [A] - p. 808

# Limites para deteccao de casos degenerados
E_MIN = 1e-4                # Excentricidade minima aceitavel
INCL_TOL = 1e-6             # Tolerancia angular (rad) para i=0 ou i=pi/2


# ============================================================================
# BLOCO 2 - FUNCOES MATEMATICAS DO METODO DE TRAGESSER
# ============================================================================


def movimento_medio(a):
    """
    Calcula o movimento medio kepleriano.

    Base teorica:
      Formula: n = sqrt(mu / a^3)
      Origem: Astrodinamica classica (3a lei de Kepler).
      Aparece implicitamente em todas as equacoes (6) e (8) do Tragesser
      e explicitamente na definicao da constante C (Apendice, p. 810).

    Entrada:
      a : float - semieixo maior [km]

    Saida:
      n : float - movimento medio [rad/s]
    """
    return math.sqrt(MU_TERRA / a**3)


def constante_C(a, n):
    """
    Calcula a constante caracteristica C do sistema tether-orbita.

    Base teorica:
      Formula: C = L * mu_m / (4 * m * n * a^4)
      Origem: Definicao no fim do Apendice de Tragesser e San (2003), p. 810.
      Todos os elementos da matriz [A] sao multiplos desta constante.

    Entradas:
      a : float - semieixo maior [km]
      n : float - movimento medio [rad/s]

    Saida:
      C : float - constante caracteristica do sistema
    """
    return (L_TETHER * MU_M) / (4.0 * M_SISTEMA * n * a**4)


def construir_matriz_A(a, e, i, omega):
    """
    Constroi a matriz [A] de dimensao 5x5 do sistema linear de guiamento.

    Base teorica:
      Origem: Apendice de Tragesser e San (2003), p. 810.
      Fundamento: cada entrada A_ij vem da substituicao da lei de corrente
      (eq. 11) nas equacoes de Gauss expandidas em serie de excentricidade
      (eq. 8) e subsequente integracao sobre um periodo orbital. Os termos
      diagonais representam o efeito direto de cada modo de corrente sobre
      seu elemento orbital correspondente; os termos fora da diagonal
      representam o acoplamento cruzado - contribuicao central do artigo.

    Entradas:
      a : float - semieixo maior inicial [km]
      e : float - excentricidade inicial [-]
      i : float - inclinacao inicial [rad]
      omega : float - argumento do perigeu inicial [rad]

    Saida:
      A : numpy.ndarray 5x5 - matriz de acoplamento do metodo
    """

    # --------------------------------------------------------------------
    # Passo 1: Calcular grandezas auxiliares
    # --------------------------------------------------------------------
    n = movimento_medio(a)
    C = constante_C(a, n)

    # Trigonometricas pre-calculadas
    cos_i = math.cos(i)
    sin_i = math.sin(i)
    cos_2w = math.cos(2.0 * omega)
    sin_2w = math.sin(2.0 * omega)

    # --------------------------------------------------------------------
    # Passo 2: Preencher a matriz A conforme o Apendice do Tragesser
    # --------------------------------------------------------------------
    A = np.zeros((5, 5))

    # Linha 1 - efeito das componentes da corrente sobre Delta a
    A[0, 0] = 8.0 * C * a * cos_i           # A_11 - corrente DC
    A[0, 1] = 8.0 * C * a * e * cos_i       # A_12 - cos(nu)
    A[0, 2] = 0.0                            # A_13 - sen(nu)
    A[0, 3] = 0.0                            # A_14 - cos[2(nu+omega)]
    A[0, 4] = 0.0                            # A_15 - sen[2(nu+omega)]

    # Linha 2 - efeito das componentes sobre Delta e
    A[1, 0] = 6.0 * C * e * cos_i
    A[1, 1] = 4.0 * C * cos_i
    A[1, 2] = 0.0
    A[1, 3] = C * e * cos_i * cos_2w
    A[1, 4] = C * e * cos_i * sin_2w

    # Linha 3 - efeito das componentes sobre Delta omega
    A[2, 0] = 4.0 * C * e * cos_i
    A[2, 1] = 0.0
    A[2, 2] = 4.0 * C * cos_i / e            # divisao por e - sensivel!
    A[2, 3] = -C * cos_i * sin_2w
    A[2, 4] = C * cos_i * (cos_2w + 1.0)

    # Linha 4 - efeito das componentes sobre Delta i
    A[3, 0] = -2.0 * C * sin_i
    A[3, 1] = 0.0
    A[3, 2] = 0.0
    A[3, 3] = -C * sin_i
    A[3, 4] = 0.0

    # Linha 5 - efeito das componentes sobre Delta Omega
    A[4, 0] = 0.0
    A[4, 1] = 0.0
    A[4, 2] = 0.0
    A[4, 3] = 0.0
    A[4, 4] = -C

    return A


def calcular_delta(orb_inicial, orb_final):
    """
    Calcula o vetor de mudancas orbitais desejadas.

    Base teorica:
      Formula: Delta = [Da, De, Dw, Di, DO]^T = orbita_final - orbita_inicial
      Origem: Definicao natural do vetor de entrada da eq. (14) do Tragesser.

    Entradas:
      orb_inicial, orb_final : dict com chaves 'a', 'e', 'omega', 'i', 'Omega'

    Saida:
      delta : numpy.ndarray de tamanho 5
    """
    delta = np.array([
        orb_final['a']     - orb_inicial['a'],
        orb_final['e']     - orb_inicial['e'],
        orb_final['omega'] - orb_inicial['omega'],
        orb_final['i']     - orb_inicial['i'],
        orb_final['Omega'] - orb_inicial['Omega'],
    ])
    return delta


def corrente_instantanea(nu, omega, X):
    """
    Avalia a lei de corrente em um ponto (nu, omega) especifico.

    Base teorica:
      Formula: I(nu, omega) = X1 + X2*cos(nu) + X3*sen(nu)
                              + X4*cos[2(nu+omega)] + X5*sen[2(nu+omega)]
      Origem: Equacao (11) de Tragesser e San (2003), p. 807.
      Interpretacao: combinacao linear das cinco formas funcionais da
      Tabela 1 do artigo, cada uma responsavel por afetar secularmente
      um elemento orbital especifico.

    Entradas:
      nu : float ou array - anomalia verdadeira [rad]
      omega : float - argumento do perigeu [rad]
      X : array de 5 elementos - coeficientes da lei de corrente [A]

    Saida:
      I : float ou array - corrente instantanea [A]
    """
    return (X[0]
            + X[1] * np.cos(nu)
            + X[2] * np.sin(nu)
            + X[3] * np.cos(2.0 * (nu + omega))
            + X[4] * np.sin(2.0 * (nu + omega)))


def calcular_I_max(X, omega, n_nu=720):
    """
    Determina o pico de corrente ao longo de uma orbita completa.

    Base teorica:
      Metodo: enumeracao discreta de I(nu) para nu em [0, 2pi], seguindo a
      recomendacao explicita de Tragesser e San (p. 807, ultimo paragrafo
      da secao Guidance Scheme): "a crude method is used in which the
      current is enumerated for a 360-deg range of true anomaly".

    Entradas:
      X : array de 5 elementos - coeficientes da lei de corrente
      omega : float - argumento do perigeu [rad]
      n_nu : int - numero de pontos na malha de nu (default 720 = 0.5 graus)

    Saida:
      I_max : float - valor maximo absoluto da corrente [A]
    """
    nu_grid = np.linspace(0.0, 2.0 * math.pi, n_nu)
    I_vals = corrente_instantanea(nu_grid, omega, X)
    return float(np.max(np.abs(I_vals)))


def resolver_guiamento_TOF_fixo(A, delta, TOF, omega):
    """
    Resolve o problema de guiamento para um TOF especificado.

    Base teorica:
      Formula: X = (1/TOF) * [A^-1] * Delta
      Origem: Equacao (15) de Tragesser e San (2003), p. 807, obtida por
      inversao direta do sistema linear [A]*t*X = Delta (equacao 14).

    Entradas:
      A : matriz 5x5 - matriz do sistema
      delta : array 5 - mudancas orbitais desejadas
      TOF : float - tempo de voo [s]
      omega : float - argumento do perigeu inicial [rad] (usado para I_max)

    Saida:
      X : array 5 - coeficientes da lei de corrente [A]
      I_max : float - pico de corrente para este TOF [A]
    """
    X = (1.0 / TOF) * np.linalg.solve(A, delta)
    I_max = calcular_I_max(X, omega)
    return X, I_max


def calcular_TOF_minimo_viavel(A, delta, omega, I_limite):
    """
    Determina o TOF minimo em que I_max iguala a corrente disponivel.

    Base teorica:
      Formula: TOF_min = TOF_ref * (I_max_ref / I_limite)
      Origem: Relacao de escala eq. (16) de Tragesser e San (2003), p. 807.
      Interpretacao: como X escala com 1/TOF (eq. 15), o pico de corrente
      tambem escala com 1/TOF. Portanto, e possivel calcular I_max para um
      TOF de referencia arbitrario e escalar linearmente para achar o TOF
      que satisfaz a restricao de hardware exatamente.

    Entradas:
      A : matriz 5x5
      delta : array 5
      omega : float - argumento do perigeu inicial [rad]
      I_limite : float - corrente maxima permitida pelo hardware [A]

    Saida:
      TOF_min : float - tempo de voo minimo viavel [s]
      X : array 5 - coeficientes correspondentes
    """
    # Passo 1: resolver para um TOF de referencia arbitrario (1 s)
    TOF_ref = 1.0
    X_ref, I_max_ref = resolver_guiamento_TOF_fixo(A, delta, TOF_ref, omega)

    # Passo 2: aplicar a relacao de escala da eq. (16)
    # Se com TOF=1 s a corrente pico e I_max_ref, entao para reduzir a
    # corrente pico a I_limite basta multiplicar TOF por I_max_ref/I_limite.
    TOF_min = TOF_ref * (I_max_ref / I_limite)

    # Passo 3: recalcular os coeficientes finais no TOF ajustado
    X = X_ref / (TOF_min / TOF_ref)

    return TOF_min, X


def detectar_casos_degenerados(orb_inicial, delta):
    """
    Verifica se a manobra pretendida cai em um dos cenarios inviaveis.

    Base teorica:
      Baseado na analise das singularidades da matriz [A] do Apendice do
      Tragesser. Casos previstos:
        1. i=0 ou i=pi (equatorial magnetica) com Delta_i ou Delta_Omega
        2. i=pi/2 (polar magnetica) com Delta_a, Delta_e ou Delta_omega
        3. e=0 (circular) - divergencia em A_33 = 4C cos(i)/e
        4. e muito baixo - mau condicionamento
        5. Fora de LEO (a > 2 R_Terra) - restricao fisica de plasma

    Entradas:
      orb_inicial : dict com elementos orbitais em SI e rad
      delta : array 5 com mudancas desejadas

    Saida:
      erros : lista de strings descrevendo os problemas detectados
              (lista vazia se tudo estiver ok)
    """
    erros = []
    i0 = orb_inicial['i']
    e0 = orb_inicial['e']
    a0 = orb_inicial['a']

    # Caso 1: equatorial magnetica com Delta_i ou Delta_Omega
    if (abs(i0) < INCL_TOL or abs(i0 - math.pi) < INCL_TOL):
        if abs(delta[3]) > 1e-9 or abs(delta[4]) > 1e-9:
            erros.append(
                "Caso 1: orbita equatorial magnetica (i=0 ou 180 graus). "
                "Impossivel alterar inclinacao ou nodo (forca perpendicular "
                "ao plano se anula)."
            )

    # Caso 2: polar magnetica com Delta_a, Delta_e ou Delta_omega
    if abs(i0 - math.pi / 2.0) < INCL_TOL:
        if (abs(delta[0]) > 1e-9 or abs(delta[1]) > 1e-9
                or abs(delta[2]) > 1e-9):
            erros.append(
                "Caso 2: orbita polar magnetica (i=90 graus). Impossivel "
                "alterar semieixo maior, excentricidade ou argumento do "
                "perigeu (forca no plano orbital se anula)."
            )

    # Caso 3: excentricidade nula
    if e0 == 0.0:
        erros.append(
            "Caso 3: excentricidade nula. Argumento do perigeu indefinido "
            "e divergencia em A_33 (divisao por zero). Utilize elementos "
            "equinociais para orbitas circulares."
        )

    # Caso 4: excentricidade muito baixa
    elif e0 < E_MIN:
        erros.append(
            f"Caso 4: excentricidade e0={e0:.2e} abaixo do limite ({E_MIN}). "
            "Matriz [A] mal condicionada; resultados podem ser imprecisos."
        )

    # Caso 5: fora do regime LEO
    if a0 > 2.0 * R_TERRA:
        erros.append(
            f"Caso 5: semieixo maior a0={a0:.1f} km fora do regime LEO. "
            "Densidade do plasma ionosferico insuficiente para gerar corrente "
            "eletrodinamica util (Tragesser e San, 2003, p. 805, Introducao)."
        )

    return erros


def kepler_nu_para_tempo(nu_alvo, e, n, t0=0.0):
    """
    Converte anomalia verdadeira em tempo desde a passagem pelo perigeu.

    Base teorica:
      Passo 1: nu -> E via tan(E/2) = sqrt((1-e)/(1+e)) * tan(nu/2)
      Passo 2: E -> M via M = E - e*sen(E)  (equacao de Kepler)
      Passo 3: M -> t via t = M/n           (definicao de anomalia media)
      Origem: astrodinamica classica; nao vem do Tragesser.

    Entradas:
      nu_alvo : array - anomalia verdadeira [rad]
      e : float - excentricidade
      n : float - movimento medio [rad/s]
      t0 : float - tempo de referencia [s]

    Saida:
      t : array - tempo correspondente [s]
    """
    E = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e))
                        * np.tan(nu_alvo / 2.0))
    M = E - e * np.sin(E)
    return t0 + M / n


def kepler_tempo_para_nu(t, e, n, tol=1e-10, max_iter=50):
    """
    Converte tempo em anomalia verdadeira via resolucao de Kepler.

    Base teorica:
      Passo 1: M = n*t                          (anomalia media)
      Passo 2: M = E - e*sen(E) -> resolver E   (Newton-Raphson)
      Passo 3: tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
      Origem: astrodinamica classica.

    Entradas:
      t : array - tempo [s]
      e : float - excentricidade
      n : float - movimento medio [rad/s]

    Saida:
      nu : array - anomalia verdadeira [rad]
    """
    M = n * t
    # Chute inicial: E = M (bom para pequenas excentricidades)
    E = M.copy() if isinstance(M, np.ndarray) else np.array([M])

    # Newton-Raphson vetorizado
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E = E - dE
        if np.max(np.abs(dE)) < tol:
            break

    nu = 2.0 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E / 2.0))
    return nu


# ============================================================================
# BLOCO 3 - INTERFACE, ENTRADA, SAIDA E GRAFICOS
# ============================================================================


def prompt_com_default(mensagem, valor_default):
    """
    Pergunta ao usuario e retorna valor default se ele apenas apertar Enter.

    Entradas:
      mensagem : str - texto da pergunta
      valor_default : float ou str - valor a usar se entrada vazia

    Saida:
      resposta : mesmo tipo do valor_default
    """
    entrada = input(f"{mensagem} [{valor_default}]: ").strip()
    if entrada == "":
        return valor_default
    if isinstance(valor_default, float):
        return float(entrada)
    if isinstance(valor_default, int):
        return int(entrada)
    return entrada


def escolher_formato_entrada():
    """
    Menu de escolha do formato de entrada dos parametros orbitais.

    Saida:
      formato : int - 1 (elementos classicos) ou 2 (altitudes)
    """
    print("\n=== Formato de entrada ===")
    print("  [1] Elementos orbitais classicos (a, e, omega, i, Omega)")
    print("  [2] Altitudes + orientacao (h_p, h_a, omega, i, Omega)")
    while True:
        opcao = prompt_com_default("Escolha", 1)
        opcao = int(opcao)
        if opcao in (1, 2):
            return opcao
        print("  Opcao invalida. Digite 1 ou 2.")


def coletar_orbita_elementos(rotulo, defaults):
    """
    Coleta os cinco elementos orbitais classicos.

    Entradas:
      rotulo : str - identificacao ('Inicial' ou 'Final')
      defaults : dict - valores padrao para cada elemento
                 (a em km, angulos em graus)

    Saida:
      orbita : dict - a[km], e[-], omega[rad], i[rad], Omega[rad]
    """
    print(f"\n=== Orbita {rotulo} (elementos classicos) ===")
    print("    Nota: angulos i e Omega no referencial magnetico.")
    a = prompt_com_default("Semieixo maior a (km)         ", defaults['a'])
    e = prompt_com_default("Excentricidade e              ", defaults['e'])
    w_deg = prompt_com_default("Argumento do perigeu w (graus)", defaults['omega'])
    i_deg = prompt_com_default("Inclinacao i (graus)          ", defaults['i'])
    O_deg = prompt_com_default("Longitude do nodo Omega (graus)", defaults['Omega'])
    return {
        'a': a,
        'e': e,
        'omega': math.radians(w_deg),
        'i': math.radians(i_deg),
        'Omega': math.radians(O_deg),
    }


def coletar_orbita_altitudes(rotulo, defaults):
    """
    Coleta parametros por altitudes (h_p, h_a) e converte para elementos.

    Base teorica:
      Formulas:
        a = R_Terra + (h_p + h_a)/2
        e = (h_a - h_p) / (2*a)
      Origem: geometria elementar da orbita eliptica.

    Entradas:
      rotulo : str
      defaults : dict com hp, ha em km e angulos em graus

    Saida:
      orbita : dict como em coletar_orbita_elementos()
    """
    print(f"\n=== Orbita {rotulo} (altitudes) ===")
    print("    Nota: angulos i e Omega no referencial magnetico.")
    hp = prompt_com_default("Altitude do perigeu h_p (km)   ", defaults['hp'])
    ha = prompt_com_default("Altitude do apogeu h_a (km)    ", defaults['ha'])
    w_deg = prompt_com_default("Argumento do perigeu w (graus)", defaults['omega'])
    i_deg = prompt_com_default("Inclinacao i (graus)          ", defaults['i'])
    O_deg = prompt_com_default("Longitude do nodo Omega (graus)", defaults['Omega'])

    a = R_TERRA + (hp + ha) / 2.0
    e = (ha - hp) / (2.0 * a)

    return {
        'a': a,
        'e': e,
        'omega': math.radians(w_deg),
        'i': math.radians(i_deg),
        'Omega': math.radians(O_deg),
    }


def escolher_modo_operacao():
    """
    Menu de escolha do modo de operacao.

    Saida:
      modo : int - 1 (pontual) ou 2 (varredura)
    """
    print("\n=== Modo de operacao ===")
    print("  [1] Calculo pontual - TOF minimo viavel dado I_disp")
    print("  [2] Varredura de TOF - como a corrente varia com o TOF")
    while True:
        opcao = int(prompt_com_default("Escolha", 1))
        if opcao in (1, 2):
            return opcao
        print("  Opcao invalida.")


def coletar_faixa_TOF():
    """
    Coleta a faixa de TOFs para varredura.

    Saida:
      TOFs : array de float - lista de TOFs em segundos
    """
    print("\n=== Faixa de TOF para varredura ===")
    TOF_min_h = prompt_com_default("TOF minimo (h)", 1.0)
    TOF_max_h = prompt_com_default("TOF maximo (h)", 500.0)
    n_pontos = int(prompt_com_default("Numero de pontos", 50))
    TOFs = np.linspace(TOF_min_h, TOF_max_h, n_pontos) * 3600.0  # h -> s
    return TOFs


def formatar_tempo(t_segundos):
    """
    Converte segundos em string 'Xh Ymin' para exibicao.
    """
    horas = int(t_segundos // 3600)
    minutos = int((t_segundos % 3600) // 60)
    return f"{horas}h {minutos}min"


def modo_dominante(X):
    """
    Identifica qual coeficiente da lei de corrente e dominante em magnitude.

    Saida:
      texto : str - descricao do modo dominante
    """
    rotulos = [
        "X_1 (DC, afeta semieixo maior)",
        "X_2 (cos nu, afeta excentricidade)",
        "X_3 (sen nu, afeta argumento do perigeu)",
        "X_4 (cos[2(nu+w)], afeta inclinacao)",
        "X_5 (sen[2(nu+w)], afeta linha dos nodos)",
    ]
    idx = int(np.argmax(np.abs(X)))
    return rotulos[idx]


def imprimir_resumo(orb_inicial, orb_final, delta, TOF, X, I_max, modo_str):
    """
    Imprime relatorio final no terminal e retorna a string do relatorio.
    """
    linhas = []
    linhas.append("")
    linhas.append("=" * 70)
    linhas.append("RESULTADO DA MANOBRA")
    linhas.append("=" * 70)
    linhas.append("")
    linhas.append(f"Modo: {modo_str}")
    linhas.append("")
    linhas.append("Orbita inicial:")
    linhas.append(f"  a = {orb_inicial['a']:.2f} km")
    linhas.append(f"  e = {orb_inicial['e']:.5f}")
    linhas.append(f"  omega = {math.degrees(orb_inicial['omega']):.3f} deg")
    linhas.append(f"  i     = {math.degrees(orb_inicial['i']):.3f} deg")
    linhas.append(f"  Omega = {math.degrees(orb_inicial['Omega']):.3f} deg")
    linhas.append("")
    linhas.append("Orbita final:")
    linhas.append(f"  a = {orb_final['a']:.2f} km")
    linhas.append(f"  e = {orb_final['e']:.5f}")
    linhas.append(f"  omega = {math.degrees(orb_final['omega']):.3f} deg")
    linhas.append(f"  i     = {math.degrees(orb_final['i']):.3f} deg")
    linhas.append(f"  Omega = {math.degrees(orb_final['Omega']):.3f} deg")
    linhas.append("")
    linhas.append("Mudancas desejadas:")
    linhas.append(f"  Delta a     = {delta[0]:+.3f} km")
    linhas.append(f"  Delta e     = {delta[1]:+.5f}")
    linhas.append(f"  Delta omega = {math.degrees(delta[2]):+.3f} deg")
    linhas.append(f"  Delta i     = {math.degrees(delta[3]):+.3f} deg")
    linhas.append(f"  Delta Omega = {math.degrees(delta[4]):+.3f} deg")
    linhas.append("")
    linhas.append("Solucao (eq. 15 de Tragesser e San, 2003):")
    linhas.append(f"  TOF = {formatar_tempo(TOF)} ({TOF/3600:.2f} h)")
    linhas.append(f"  I_max = {I_max:.4f} A  (I_disponivel = {I_DISP} A)")
    linhas.append("")
    linhas.append("Coeficientes da lei de corrente (eq. 11):")
    linhas.append(f"  X_1 (DC)              = {X[0]:+.4f} A")
    linhas.append(f"  X_2 (cos nu)          = {X[1]:+.4f} A")
    linhas.append(f"  X_3 (sen nu)          = {X[2]:+.4f} A")
    linhas.append(f"  X_4 (cos[2(nu+w)])    = {X[3]:+.4f} A")
    linhas.append(f"  X_5 (sen[2(nu+w)])    = {X[4]:+.4f} A")
    linhas.append("")
    linhas.append(f"Modo dominante: {modo_dominante(X)}")
    linhas.append("=" * 70)
    linhas.append("")

    texto = "\n".join(linhas)
    print(texto)
    return texto


# --------- Geracao dos graficos ---------

def grafico_I_nu(X, omega, pasta):
    """
    Grafico 1: perfil da corrente ao longo de uma orbita (I vs nu).
    Analogo as Figuras 4 e 5 do Tragesser (mas em funcao de nu, nao de t).
    """
    nu = np.linspace(0.0, 2.0 * math.pi, 720)
    I = corrente_instantanea(nu, omega, X)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.degrees(nu), I, color='#1f4e79', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("Anomalia verdadeira nu (graus)")
    ax.set_ylabel("Corrente I (A)")
    ax.set_title("Perfil da corrente ao longo de uma orbita completa")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    caminho = os.path.join(pasta, "grafico_1_I_nu.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {caminho}")


def grafico_I_tempo(X, orbita, TOF, pasta, n_orbitas=3):
    """
    Grafico 2: perfil da corrente no tempo (I vs t) ao longo de N orbitas.

    Utiliza cinematica kepleriana pura (nao integra Gauss) para mapear
    nu(t) - hipotese consistente com o metodo de Tragesser (elementos
    orbitais aproximadamente constantes durante a integracao).
    """
    a = orbita['a']
    e = orbita['e']
    omega = orbita['omega']
    n = movimento_medio(a)
    T_orbital = 2.0 * math.pi / n  # periodo orbital [s]

    # Duracao: N orbitas ou TOF completo, o que for menor
    t_max = min(n_orbitas * T_orbital, TOF)
    t = np.linspace(0.0, t_max, 2000)
    nu = kepler_tempo_para_nu(t, e, n)
    I = corrente_instantanea(nu, omega, X)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t / 3600.0, I, color='#0f6e2e', linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("Tempo (h)")
    ax.set_ylabel("Corrente I (A)")
    ax.set_title(f"Perfil temporal da corrente ({n_orbitas} orbitas iniciais)")
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    caminho = os.path.join(pasta, "grafico_2_I_t.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {caminho}")


def grafico_Imax_vs_TOF(TOFs, I_max_list, TOF_critico, pasta):
    """
    Grafico 3: I_max em funcao do TOF (varredura).
    Marca o TOF critico onde I_max = I_disp.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(np.array(TOFs) / 3600.0, I_max_list,
              color='#a11d1d', linewidth=1.8, marker='o', markersize=3)
    ax.axhline(I_DISP, color='black', linestyle='--', linewidth=1,
               label=f"I_disp = {I_DISP} A")
    if TOF_critico is not None:
        ax.axvline(TOF_critico / 3600.0, color='blue', linestyle=':',
                   linewidth=1, label=f"TOF critico = {TOF_critico/3600:.1f} h")
    ax.set_xlabel("TOF (h)")
    ax.set_ylabel("I_max necessaria (A)")
    ax.set_title("Corrente pico exigida em funcao do tempo de voo (eq. 15)")
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    caminho = os.path.join(pasta, "grafico_3_Imax_vs_TOF.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {caminho}")


def grafico_Imax_vs_TOF_linear(TOFs, I_max_list, TOF_critico, pasta):
    """
    Grafico 7: I_max em funcao do TOF em escala LINEAR.

    Base teorica:
      A relacao I_max = k/TOF, derivada da equacao (15) de Tragesser e San
      (2003), aparece como uma reta em escala log-log (grafico 3) e como
      uma hiperbole em escala linear (este grafico). Ambas as visualizacoes
      representam o mesmo dado numerico; a versao linear e util para
      comunicacao intuitiva da relacao inversamente proporcional entre
      tempo de voo e corrente exigida.

    Entradas:
      TOFs : array - tempos de voo varridos [s]
      I_max_list : array - correntes maximas correspondentes [A]
      TOF_critico : float ou None - TOF em que I_max = I_disp [s]
      pasta : str - caminho da pasta de saida
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    TOFs_h = np.array(TOFs) / 3600.0
    ax.plot(TOFs_h, I_max_list,
            color='#a11d1d', linewidth=1.8, marker='o', markersize=3,
            label="I_max necessaria")
    ax.axhline(I_DISP, color='black', linestyle='--', linewidth=1,
               label=f"I_disp = {I_DISP} A")
    if TOF_critico is not None:
        ax.axvline(TOF_critico / 3600.0, color='blue', linestyle=':',
                   linewidth=1,
                   label=f"TOF critico = {TOF_critico/3600:.1f} h")

    # Destacar visualmente a regiao inviavel (acima de I_disp)
    ax.fill_between(TOFs_h, I_DISP, np.max(I_max_list) * 1.05,
                    where=(np.array(I_max_list) > I_DISP),
                    color='red', alpha=0.08, label="Regiao inviavel")

    ax.set_xlabel("TOF (h)")
    ax.set_ylabel("I_max necessaria (A)")
    ax.set_title("Corrente pico vs. tempo de voo (escala linear)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    # Limitar eixo y para nao esmagar a hiperbole com pontos muito altos
    y_max_plot = min(np.max(I_max_list), 5.0 * I_DISP)
    ax.set_ylim(0, y_max_plot)
    ax.set_xlim(0, TOFs_h[-1])

    fig.tight_layout()
    caminho = os.path.join(pasta, "grafico_7_Imax_vs_TOF_linear.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {caminho}")


def grafico_Xi_vs_TOF(TOFs, X_matrix, pasta):
    """
    Grafico 4: coeficientes X_1..X_5 em funcao do TOF (varredura).
    """
    rotulos = ["X_1 (DC)", "X_2 (cos nu)", "X_3 (sen nu)",
               "X_4 (cos[2(nu+w)])", "X_5 (sen[2(nu+w)])"]
    cores = ['#1f4e79', '#0f6e2e', '#a11d1d', '#7d3c98', '#c67200']

    fig, ax = plt.subplots(figsize=(9, 5))
    TOFs_h = np.array(TOFs) / 3600.0
    for k in range(5):
        ax.plot(TOFs_h, X_matrix[:, k], color=cores[k],
                linewidth=1.4, label=rotulos[k])
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel("TOF (h)")
    ax.set_ylabel("Coeficiente X_i (A)")
    ax.set_title("Decomposicao dos coeficientes da lei de corrente")
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    caminho = os.path.join(pasta, "grafico_4_Xi_vs_TOF.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {caminho}")


def grafico_barras_Xi(X, pasta):
    """
    Grafico 5: |X_i| como barras verticais para o TOF escolhido.
    Mostra visualmente qual componente da corrente domina.
    """
    rotulos = ["X_1\n(DC)", "X_2\n(cos)", "X_3\n(sen)",
               "X_4\n(cos 2)", "X_5\n(sen 2)"]
    cores = ['#1f4e79', '#0f6e2e', '#a11d1d', '#7d3c98', '#c67200']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rotulos, np.abs(X), color=cores)
    ax.set_ylabel("|X_i| (A)")
    ax.set_title("Magnitude dos coeficientes da lei de corrente")
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    caminho = os.path.join(pasta, "grafico_5_barras_Xi.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {caminho}")


def _pontos_orbita_3d(orbita, n_pontos=360):
    """
    Retorna as coordenadas (x, y, z) da orbita no espaco inercial.

    Base teorica:
      Formulas: r(nu) = a(1-e^2)/(1+e*cos nu)
                Posicao no plano orbital: (r cos nu, r sen nu, 0)
                Rotacao para o inercial: R_z(Omega) * R_x(i) * R_z(omega)
      Origem: astrodinamica classica (Vallado, Fundamentals of
      Astrodynamics, cap. 2).
    """
    a = orbita['a']
    e = orbita['e']
    w = orbita['omega']
    i = orbita['i']
    O = orbita['Omega']

    nu = np.linspace(0.0, 2.0 * math.pi, n_pontos)
    r = a * (1.0 - e**2) / (1.0 + e * np.cos(nu))

    # Posicao no plano orbital (perigeu no eixo x local)
    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)
    z_p = np.zeros_like(nu)

    # Rotacoes: R_z(omega), depois R_x(i), depois R_z(Omega)
    cos_w, sin_w = math.cos(w), math.sin(w)
    cos_i, sin_i = math.cos(i), math.sin(i)
    cos_O, sin_O = math.cos(O), math.sin(O)

    # R_z(w) aplicada
    x1 = cos_w * x_p - sin_w * y_p
    y1 = sin_w * x_p + cos_w * y_p
    z1 = z_p
    # R_x(i) aplicada
    x2 = x1
    y2 = cos_i * y1 - sin_i * z1
    z2 = sin_i * y1 + cos_i * z1
    # R_z(Omega) aplicada
    x = cos_O * x2 - sin_O * y2
    y = sin_O * x2 + cos_O * y2
    z = z2

    return x, y, z


def grafico_orbitas_3d(orb_inicial, orb_final, pasta):
    """
    Grafico 6: orbitas inicial e final no espaco 3D com a Terra na origem.
    """
    xi, yi, zi = _pontos_orbita_3d(orb_inicial)
    xf, yf, zf = _pontos_orbita_3d(orb_final)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xi, yi, zi, color='#1f4e79', linewidth=1.6, label='Orbita inicial')
    ax.plot(xf, yf, zf, color='#a11d1d', linewidth=1.6, label='Orbita final')

    # Esfera representando a Terra
    u, v = np.meshgrid(np.linspace(0, 2 * math.pi, 40),
                       np.linspace(0, math.pi, 20))
    xs = R_TERRA * np.cos(u) * np.sin(v)
    ys = R_TERRA * np.sin(u) * np.sin(v)
    zs = R_TERRA * np.cos(v)
    ax.plot_surface(xs, ys, zs, color='lightblue', alpha=0.3,
                    edgecolor='none')

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title("Orbitas inicial e final no referencial inercial magnetico")
    ax.legend()

    # Aspecto igual nos tres eixos
    max_range = max(np.ptp(np.concatenate([xi, xf])),
                    np.ptp(np.concatenate([yi, yf])),
                    np.ptp(np.concatenate([zi, zf]))) / 2.0
    mid_x = (np.concatenate([xi, xf]).max() + np.concatenate([xi, xf]).min()) / 2
    mid_y = (np.concatenate([yi, yf]).max() + np.concatenate([yi, yf]).min()) / 2
    mid_z = (np.concatenate([zi, zf]).max() + np.concatenate([zi, zf]).min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    fig.tight_layout()
    caminho = os.path.join(pasta, "grafico_6_orbitas_3d.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {caminho}")


def salvar_csv_varredura(TOFs, X_matrix, I_max_list, pasta):
    """
    Salva a tabela da varredura de TOF em formato CSV.
    """
    caminho = os.path.join(pasta, "varredura_tof.csv")
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["TOF_h", "X1_A", "X2_A", "X3_A", "X4_A", "X5_A", "Imax_A"])
        for k, TOF in enumerate(TOFs):
            w.writerow([f"{TOF/3600:.4f}",
                        f"{X_matrix[k, 0]:.6f}",
                        f"{X_matrix[k, 1]:.6f}",
                        f"{X_matrix[k, 2]:.6f}",
                        f"{X_matrix[k, 3]:.6f}",
                        f"{X_matrix[k, 4]:.6f}",
                        f"{I_max_list[k]:.6f}"])
    print(f"  Salvo: {caminho}")


def salvar_resumo_txt(texto, pasta):
    """
    Salva o resumo textual em um arquivo .txt.
    """
    caminho = os.path.join(pasta, "resumo.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"  Salvo: {caminho}")


def criar_pasta_resultados():
    """
    Cria uma subpasta 'resultados/YYYY-MM-DD_HH-MM-SS' ao lado do script.
    """
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "resultados")
    os.makedirs(base, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pasta = os.path.join(base, stamp)
    os.makedirs(pasta, exist_ok=True)
    return pasta


# ============================================================================
# BLOCO 4 - PROGRAMA PRINCIPAL
# ============================================================================


# Defaults do caso "Orbit Raising" do Tragesser (Secao Results, p. 808).
# Se o usuario apertar Enter em todas as perguntas, reproduz esse caso -
# e os resultados devem se aproximar dos valores da equacao (19).
DEFAULTS_INICIAL_ELEM = {
    'a': 6878.0,
    'e': 0.02,
    'omega': 30.0,
    'i': 30.0,
    'Omega': 50.0,
}
DEFAULTS_FINAL_ELEM = {
    'a': 7078.0,      # a_inicial + 200 km (Delta_a do exemplo do artigo)
    'e': 0.02,
    'omega': 30.0,
    'i': 30.0,
    'Omega': 50.0,
}
DEFAULTS_INICIAL_ALT = {
    'hp': 6878.0 - R_TERRA - (0.02 * 6878.0),   # h_p da orbita inicial
    'ha': 6878.0 - R_TERRA + (0.02 * 6878.0),   # h_a da orbita inicial
    'omega': 30.0,
    'i': 30.0,
    'Omega': 50.0,
}
DEFAULTS_FINAL_ALT = {
    'hp': 7078.0 - R_TERRA - (0.02 * 7078.0),
    'ha': 7078.0 - R_TERRA + (0.02 * 7078.0),
    'omega': 30.0,
    'i': 30.0,
    'Omega': 50.0,
}


def main():
    """
    Fluxo principal do programa.
    """
    print("=" * 70)
    print("SIMULACAO DE MANOBRA ORBITAL COM TETHER ELETRODINAMICO")
    print("Camada de Guiamento - Metodo de Tragesser e San (2003)")
    print("=" * 70)
    print("\nPressione Enter em qualquer pergunta para usar o valor padrao")
    print("(que reproduz o exemplo de 'orbit raising' do artigo, eq. 19).")

    # -------- Etapa 1: formato de entrada --------
    formato = escolher_formato_entrada()

    # -------- Etapa 2: coleta das orbitas --------
    if formato == 1:
        orb_inicial = coletar_orbita_elementos("Inicial", DEFAULTS_INICIAL_ELEM)
        orb_final = coletar_orbita_elementos("Final", DEFAULTS_FINAL_ELEM)
    else:
        orb_inicial = coletar_orbita_altitudes("Inicial", DEFAULTS_INICIAL_ALT)
        orb_final = coletar_orbita_altitudes("Final", DEFAULTS_FINAL_ALT)

    # -------- Etapa 3: mudancas orbitais desejadas --------
    delta = calcular_delta(orb_inicial, orb_final)

    # -------- Etapa 4: verificacao de casos degenerados --------
    erros = detectar_casos_degenerados(orb_inicial, delta)
    if erros:
        print("\n" + "!" * 70)
        print("MANOBRA INVIAVEL - problemas detectados:")
        for erro in erros:
            print(f"  - {erro}")
        print("!" * 70)
        return

    # -------- Etapa 5: montagem da matriz [A] --------
    A = construir_matriz_A(orb_inicial['a'], orb_inicial['e'],
                           orb_inicial['i'], orb_inicial['omega'])

    # -------- Etapa 6: escolha do modo de operacao --------
    modo = escolher_modo_operacao()

    # Pasta de saida
    pasta = criar_pasta_resultados()
    print(f"\nResultados serao salvos em: {pasta}\n")

    # -------- Etapa 7A: modo pontual --------
    if modo == 1:
        TOF, X = calcular_TOF_minimo_viavel(A, delta,
                                            orb_inicial['omega'], I_DISP)
        I_max = calcular_I_max(X, orb_inicial['omega'])
        modo_str = "Calculo pontual (TOF minimo viavel para I_disp)"

        texto = imprimir_resumo(orb_inicial, orb_final, delta,
                                TOF, X, I_max, modo_str)

        # Graficos aplicaveis ao modo pontual
        print("Gerando graficos...")
        grafico_I_nu(X, orb_inicial['omega'], pasta)
        grafico_I_tempo(X, orb_inicial, TOF, pasta)
        grafico_barras_Xi(X, pasta)
        grafico_orbitas_3d(orb_inicial, orb_final, pasta)

        salvar_resumo_txt(texto, pasta)

    # -------- Etapa 7B: modo varredura --------
    else:
        TOFs = coletar_faixa_TOF()
        n = len(TOFs)
        X_matrix = np.zeros((n, 5))
        I_max_list = np.zeros(n)

        for k, TOF in enumerate(TOFs):
            X_k, Imax_k = resolver_guiamento_TOF_fixo(
                A, delta, TOF, orb_inicial['omega']
            )
            X_matrix[k, :] = X_k
            I_max_list[k] = Imax_k

        # Localizar TOF critico (onde I_max = I_disp)
        TOF_critico = None
        if I_max_list[0] > I_DISP > I_max_list[-1]:
            # Interpolacao linear em log-log para achar o cruzamento
            log_TOF = np.log(TOFs)
            log_I = np.log(I_max_list)
            log_Idisp = math.log(I_DISP)
            idx = np.where(np.diff(np.sign(log_I - log_Idisp)))[0][0]
            frac = ((log_Idisp - log_I[idx])
                    / (log_I[idx + 1] - log_I[idx]))
            log_TOF_c = log_TOF[idx] + frac * (log_TOF[idx + 1] - log_TOF[idx])
            TOF_critico = math.exp(log_TOF_c)

        # Para o resumo, usar o TOF critico (ou o mediano se nao houver)
        if TOF_critico is not None:
            TOF_ref = TOF_critico
        else:
            TOF_ref = TOFs[n // 2]
        X_ref, Imax_ref = resolver_guiamento_TOF_fixo(
            A, delta, TOF_ref, orb_inicial['omega']
        )
        modo_str = f"Varredura de TOF ({TOFs[0]/3600:.1f} h a "\
                   f"{TOFs[-1]/3600:.1f} h, {n} pontos)"

        texto = imprimir_resumo(orb_inicial, orb_final, delta,
                                TOF_ref, X_ref, Imax_ref, modo_str)

        print("Gerando graficos e tabelas...")
        grafico_I_nu(X_ref, orb_inicial['omega'], pasta)
        grafico_I_tempo(X_ref, orb_inicial, TOF_ref, pasta)
        grafico_Imax_vs_TOF(TOFs, I_max_list, TOF_critico, pasta)
        grafico_Imax_vs_TOF_linear(TOFs, I_max_list, TOF_critico, pasta)
        grafico_Xi_vs_TOF(TOFs, X_matrix, pasta)
        grafico_barras_Xi(X_ref, pasta)
        grafico_orbitas_3d(orb_inicial, orb_final, pasta)

        salvar_csv_varredura(TOFs, X_matrix, I_max_list, pasta)
        salvar_resumo_txt(texto, pasta)

    print("\nExecucao concluida com sucesso.\n")


if __name__ == "__main__":
    main()