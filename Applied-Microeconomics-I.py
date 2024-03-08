# %% KÜNYE

''' 
3. COURSECON International Summer Seminars - 2023 July & August

Genç Ekonomistler Kulübü

Lecture: Prof. Mustafa Akal / Sakarya University

Create (Applied) : Yasin Tosun / Siegen University 

Assistant : Muhammed Eyüp Gökçek (BSc) / Dokuz Eylul University

'''
# %% Course Application

###############################################################################
# 1. Condition of Necessity in the Optimization Process
###############################################################################



" y(x) = ax^2 - bx "

import numpy as np
import matplotlib.pyplot as plt
# Function definition
def y(x, a, b):
    return a * x ** 2 - b * x
# Variable range
x_aralığı = np.linspace(-10, 10, 100)
# Parameters
a = 2
b = 1
# Calculate function
y_değerleri = y(x_aralığı, a, b)
# Draw graph
plt.plot(x_aralığı, y_değerleri)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y(x) = ax^2 - bx')
plt.grid(True)
plt.show()

" y(x) = ax^3 - bx^2 + cx "

import numpy as np
import matplotlib.pyplot as plt
# Function definition
def y(x, a, b, c):
    return a * x ** 3 - b * x ** 2 + c * x
# Variable range
x_aralığı = np.linspace(-10, 10, 100)
# Parameters
a = 1
b = 2
c = 1
# Calculate function
y_değerleri = y(x_aralığı, a, b, c)
# Draw graph
plt.plot(x_aralığı, y_değerleri)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y(x) = ax^3 - bx^2 + cx')
plt.grid(True)
plt.show()

" y(x) = -ax^2 + bx "

import numpy as np
import matplotlib.pyplot as plt
# Function definition
def y(x, a, b):
    return -a * x ** 2 + b * x
# Variable range
x_aralığı = np.linspace(-10, 10, 100)
# Parameters
a = 1
b = 2
# Calculate function
y_değerleri = y(x_aralığı, a, b)
# Draw graph
plt.plot(x_aralığı, y_değerleri)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y(x) = -ax^2 + bx')
plt.grid(True)
plt.show()


###############################################################################
# 2. Increasing and Decreasing Functions and Optimum Value
###############################################################################



" fonksiyon artan olduğunda : y(x) = 2x + 3 "

import numpy as np
import matplotlib.pyplot as plt
# Function definition
def y(x):
    return 2 * x + 3
# Variable range
x_aralığı = np.linspace(-10, 10, 100)
# Calculate Function
y_değerleri = y(x_aralığı)
# Draw Graph
plt.plot(x_aralığı, y_değerleri)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y(x) = 2x + 3')
plt.grid(True)
plt.show()


" fonksiyon azalan olduğunda : y(x) = -0.5x + 2 "

import numpy as np
import matplotlib.pyplot as plt
# Function definition
def y(x):
    return -0.5 * x + 2
# Variable range
x_aralığı = np.linspace(-10, 10, 100)
# Fonksiyonun hesaplanması
y_değerleri = y(x_aralığı)
# Grafiğin çizimi
plt.plot(x_aralığı, y_değerleri)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y(x) = -0.5x + 2')
plt.grid(True)
plt.show()


###############################################################################
# 3. Konkavlık
###############################################################################



" aşağı doğru konkav : y(x) = -0.5x^2 + 2x + 3 "

import numpy as np
import matplotlib.pyplot as plt
# Function definition
def y(x):
    return -0.5 * x ** 2 + 2 * x + 3
# Variable range
x_aralığı = np.linspace(-10, 10, 100)
# Fonksiyonun hesaplanması
y_değerleri = y(x_aralığı)
# Grafiğin çizimi
plt.plot(x_aralığı, y_değerleri)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y(x) = -0.5x^2 + 2x + 3')
plt.grid(True)
plt.show()

" yukarı doğru konkav : y(x) = 0.5x^2 - 2x - 3 "

import numpy as np
import matplotlib.pyplot as plt
# Function definition
def y(x):
    return 0.5 * x ** 2 - 2 * x - 3
# Variable range
x_aralığı = np.linspace(-10, 10, 100)
# Fonksiyonun hesaplanması
y_değerleri = y(x_aralığı)
# Grafiğin çizimi
plt.plot(x_aralığı, y_değerleri)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y(x) = 0.5x^2 - 2x - 3')
plt.grid(True)
plt.show()


###############################################################################
# 4. Eşanlı Denge
###############################################################################


import numpy as np
import matplotlib.pyplot as plt
# Talep ve arz fonksiyonları
def talep_fonksiyonu(fiyat):
    return 100 - 2 * fiyat
def arz_fonksiyonu(fiyat):
    return 10 + 3 * fiyat
# Fiyat aralığı
fiyat_aralığı = np.linspace(0, 30, 100)
# Talep ve arz miktarlarının hesaplanması
talep_miktarları = talep_fonksiyonu(fiyat_aralığı)
arz_miktarları = arz_fonksiyonu(fiyat_aralığı)
# Grafiğin çizimi
plt.plot(fiyat_aralığı, talep_miktarları, label='Talep')
plt.plot(fiyat_aralığı, arz_miktarları, label='Arz')
plt.xlabel('Fiyat')
plt.ylabel('Miktar')
plt.legend()
plt.title('Mikroiktisatta Eş Anlı Denge')
plt.grid(True)
plt.show()



###############################################################################
# 5. Talep Fonksiyonu - 1
###############################################################################


" TALEP FONKSİYONU - 1 "

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve

# Değişkenler
P, Q_D, I, P_T, P_I = symbols('P Q_D I P_T P_I')

# Talep fonksiyonu
demand_equation = Eq(Q_D, -4*P + 0.01*I - 5*P_T + 10*P_I)

# Fiyatı bulma
solutions = solve(demand_equation, P)
P_value = solutions[0] if len(solutions) > 0 else None
print("Fiyat değeri:", P_value)

# Talep miktarını hesaplama
demand_function = demand_equation.rhs
Q_D_values = [demand_function.subs({P: price, I: 8000, P_T: 8, P_I: 4}) for price in np.linspace(0, 10, 100)]

# Grafiği çizme
plt.plot(np.linspace(0, 10, 100), Q_D_values)
plt.xlabel('Fiyat (P)')
plt.ylabel('Talep (Q_D)')
plt.title('Talep Denklemi')
plt.grid(True)

plt.show()


###############################################################################
# 6. Talep Fonksiyonu - 2 : P_I = 8 olunca ne olur ?
###############################################################################


" TALEP FONKSİYONU - 2 : P_I = 8 olunca ne olur ? "

# Değişkenler
P, Q_D, I, P_T, P_I = symbols('P Q_D I P_T P_I')

# Talep fonksiyonu
demand_equation = Eq(Q_D, -4*P + 0.01*I - 5*P_T + 10*P_I)

# Fiyatı bulma
solutions = solve(demand_equation, P)
P_value = solutions[0] if len(solutions) > 0 else None
print("Fiyat değeri:", P_value)

# P_I değerini güncelleme
P_I_value = 8

# Talep miktarını hesaplama
demand_function = demand_equation.rhs
Q_D_values = [demand_function.subs({P: price, I: 8000, P_T: 8, P_I: P_I_value}) for price in np.linspace(0, 10, 100)]

# Grafiği Çizme
plt.plot(np.linspace(0, 10, 100), Q_D_values)
plt.xlabel('Fiyat (P)')
plt.ylabel('Talep (Q_D)')
plt.title('Talep Denklemi (P_I = 8)')
plt.grid(True)

plt.show()


###############################################################################
# 7. Rasyonel Seçimler - 1 : Karşılaştırma ( Bütünlük , Tamamlayıcılık)
###############################################################################


" Tamamlayıcılık : A > B "

def utility(A, B):
    return A + B

# Mal demeti A ve B
A = 9
B = 5

# Mal demeti A'nın toplam faydası
fayda_A = utility(A, B)
print(fayda_A)

# Mal demeti B'nin toplam faydası (B'yi artık A'ya göre daha az çekici yapalım)
fayda_B = utility(A - 2, B)
print(fayda_B)

# Karşılaştırma
if fayda_A > fayda_B:
    print("Birey A'yı B'ye tercih eder.")
elif fayda_A < fayda_B:
    print("Birey B'yi A'ya tercih eder.")
else:
    print("Birey A ve B arasında tercih yapmaz.")
    

" Tamamlayıcılık : B > A "

def utility(A, B):
    return A + B

# Mal demeti A ve B
A = 9
B = 5

# Mal demeti A'nin toplam faydası (A'yı artık B'yi göre daha az çekici yapalım)
fayda_A = utility(A, B - 1)
print(fayda_A)

# Mal demeti B'nin toplam faydası 
fayda_B = utility(A, B)
print(fayda_B)

# Karşılaştırma
if fayda_A > fayda_B:
    print("Birey A'yı B'ye tercih eder.")
elif fayda_A < fayda_B:
    print("Birey B'yi A'ya tercih eder.")
else:
    print("Birey A ve B arasında tercih yapmaz.")
    
    
" Tamamlayıcılık : A = B "

def utility(A, B):
    return A + B

# Mal demeti A ve B
A = 9
B = 5

# Mal demeti A'nın toplam faydası
fayda_A = utility(A, B)
print(fayda_A)

# Mal demeti B'nin toplam faydası
fayda_B = utility(B, A)
print(fayda_B)

# Karşılaştırma
if fayda_A > fayda_B:
    print("Birey A'yı B'ye tercih eder.")
elif fayda_A < fayda_B:
    print("Birey B'yi A'ya tercih eder.")
else:
    print("Birey A ve B arasında tercih yapmaz.")
    
# 8. Fayda

" Marjinal Fayda "

def marjinal_fayda(fayda, miktar):
    """
    Basit bir marjinal fayda fonksiyonu.
    Parametre olarak fayda fonksiyonu ve miktar alır.
    Marjinal faydayı hesaplar.
    """
    return fayda(miktar) - fayda(miktar - 1) if miktar > 0 else fayda(1)

def fayda(miktar):
    """
    Basit bir fayda fonksiyonu.
    Parametre olarak miktarı alır ve miktarın karesini döndürür.
    """
    return miktar ** 2

# Örnek kullanım
miktar = 5

mf = marjinal_fayda(fayda, miktar)
print("Marjinal Fayda:", mf)

###############################################################################
# 9. Fayda - 2 : Marjinal, Toplam Fayda ve Azalan Marjinal Fayda İlkesi
###############################################################################

" ÖRNEK : U = f(q) = 20q – q^2 "


from sympy import symbols, Eq, solve

def total_utility(quantity):
    return 20 * quantity - quantity**2

def marginal_utility(quantity):
    return 20 - 2 * quantity

# Doyum Noktası
quantity = symbols('quantity')
equation = Eq(marginal_utility(quantity), 0)
solution = solve(equation, quantity)
doyum_noktasi = solution[0]

# Maksimum Toplam Fayda
maksimum_fayda = total_utility(10)

# Azalan Marjinal Fayda İlkesi
azalan_marjinal_fayda_ilkesi = True

print("Doyum Noktası:", doyum_noktasi)
print("Maksimum Toplam Fayda:", maksimum_fayda)
print("Azalan Marjinal Fayda İlkesi Uyumu:", azalan_marjinal_fayda_ilkesi)

import numpy as np
import matplotlib.pyplot as plt

def total_utility(quantity):
    return 20 * quantity - quantity**2

quantity = np.linspace(0, 20, 100)  # Mal miktarı aralığı
utility = total_utility(quantity)

plt.plot(quantity, utility)
plt.axvline(x=10, color='red', linestyle='--', label='Doyum Noktası')
plt.xlabel("Quantity")
plt.ylabel("Total Utility")
plt.title("Total Utility Function")
plt.legend()
plt.grid(True)
plt.show()

###############################################################################
# 10. Marjinal İkame Oranı
###############################################################################

" Marjinal İkame Oranı : U = 4 * X * Y**2 ==> X=30 Y=48 "

import sympy as sp

# Tüketim bileşimleri
X = sp.Symbol('X')
Y = sp.Symbol('Y')

# Fayda fonksiyonu
U = 4 * X * Y**2

# X'nin marjinal faydası (MFX)
MFX = sp.diff(U, X)

# Y'nin marjinal faydası (MFY)
MFY = sp.diff(U, Y)

# Marjinal ikame oranı (MRS)
MRS = MFX / MFY

# Tüketim bileşimleri yerine değerleri yerleştirme
MRS_val = MRS.subs([(X, 30), (Y, 48)])

print("Marjinal İkame Oranı (MRS):", MRS_val)



###############################################################################
# 11. La-Grange Yöntemi ile Fayda Maksimizasyonu
###############################################################################

" A. Tek Mal Durumu "

from sympy import symbols, diff, solve

# Sembolik değişkenleri tanımlayın
X, P_x = symbols('X P_x')

# Fayda fonksiyonunu tanımlayın
X, P_x = symbols('X P_x')
U = 20 * X - X**2

# Bütçe kısıtını tanımlayın
X, P_x = symbols('X P_x')
I = 100
B = I - P_x * X

# Fayda fonksiyonunu maksimize eden X miktarını ve P_x fiyatını bulun
solution = solve((diff(U, X), B), (X, P_x))

# Optimum değerleri elde edin
optimal_X = solution[0][0]
optimal_P_x = solution[0][1]

print("Optimum X değeri:", optimal_X)
print("Optimum P_x değeri:", optimal_P_x)

from sympy import symbols, diff

# Sembolik değişkenleri tanımlayın
X = symbols('X')

# Fayda fonksiyonunu tanımlayın
U = 20 * X - X**2

# Fayda fonksiyonunun ikinci türevini hesaplayın
U_xx = diff(U, X, 2)

# Optimum değerleri kullanarak ikinci türevi değerlendirin
Hessian_value = U_xx.subs(X, optimal_X)

print("Hessian Matris Değeri : ", Hessian_value)

# Hessian matrisini kontrol edin
if Hessian_value > 0:
    optimal_fayda = U.subs(X, optimal_X)
    print("Optimum değerde minimum fayda elde edilir. Global minimum vardır.")
    print("Optimum değerde minimum fayda:", optimal_fayda)
    
elif Hessian_value < 0:
    optimal_fayda = U.subs(X, optimal_X)
    print("Optimum değerde maksimum fayda elde edilir. Global maksimum vardır.")
    print("Optimum değerde maksimum fayda:", optimal_fayda)
else:
    print("Optimum değerde kararsızlık durumu vardır.")


###############################################################################
# 12. Esneklik
###############################################################################

" F. 1. Talebin Fiyat Esnekliği "

def fiyat_esnekligi(talep_miktarlari, fiyatlar):
    esneklik = (talep_miktarlari[1] - talep_miktarlari[0]) / (fiyatlar[1] - fiyatlar[0])
    return esneklik

talep_miktarlari = [10, 8, 6, 4, 2]
fiyatlar = [5, 10, 15, 20, 25]

esneklik = fiyat_esnekligi(talep_miktarlari, fiyatlar)

if esneklik < 1:
    print("Fiyat Esnekliği:", esneklik ,"Esnek")
else:
    print("Fiyat Esnekliği:", esneklik, "İnelastik")



###############################################################################
# 13. Marjinal Verim
###############################################################################

from sympy import symbols, diff

L = symbols('L')  # Emek miktarını temsil eden sembol

Q = 10*L - 0.5*L**2  # Üretim fonksiyonu

marginal_verim = diff(Q, L)  # Marjinal verim hesaplama

print("Marjinal Verim:", marginal_verim)


###############################################################################
# 14. MPL + APL + TPL
###############################################################################

" ÖRNEK : Q = 90*(K**2)*(L**2) - (K**3)*(L**3) "

"A. MPL ve TPL'nin Max olduğu L seviyesi"

from sympy import symbols, diff , solve

L = symbols('L')
Q = 360 * L ** 2 - 8 * L ** 3

MPL_expr = diff(Q, L)

print("MPL ifadesi:", MPL_expr)

K = 2
L = symbols('L')
MPL_expr = -3 * K ** 3 * L ** 2 + 180 * K ** 2 * L

solution = solve(MPL_expr, L)

L_val = [sol.evalf() for sol in solution if sol > 0]

print("optimal L2 değeri:", L_val)

" B. APL ve APL nin max olduğu L seviyesi"

from sympy import symbols, diff, solve

L = symbols('L')
K = 2

Q_expr = 360 * (L ** 2) - 8 * (L ** 3)
AP_expr = Q_expr / L

dAP = diff(AP_expr, L)
optimal_L = solve(dAP, L)[0]

print("Optimal L1 değeri:", optimal_L)

" C. MPL ve MPL ile TPL nin max olduğu L seviyesi "

from sympy import symbols, diff, solve

L = symbols('L')
K = 2

MPL_expr = -3 * (K ** 3) * (L ** 2) + 180 * (K ** 2) * L

dMPL = diff(MPL_expr, L)
optimal_L = solve(dMPL, L)[0]

print("Optimal L0 değeri:", optimal_L)

###############################################################################
# 15. Marjinal Teknik İkame Oranı
###############################################################################

def marjinal_teknik_ikame_orani(MPL, MPK):
    MRTS = MPL / MPK
    return MRTS

K = 2
L = 3

MPL = 180 * (K**2) * L - 3 * (K ** 3) * (L**2)
MPK = 180 * K * (L ** 2) - 3 * (K ** 2) * (L ** 3)

MRTS = marjinal_teknik_ikame_orani(MPL, MPK)

print("Marjinal Teknik İkame Oranı:", MRTS)

###############################################################################
# This is just a fragment of what we can do...
###############################################################################