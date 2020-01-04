import sys

## ./casa -c /home/yoyisaurio/Desktop/proyecto/deMSaTXT.py /home/yoyisaurio/Desktop/HLTau_B6cont.calavg.tav300s

# # Apertura de datos y cabecera de datos
# ms.open("/home/yoyisaurio/Desktop/HLTau_B6cont.calavg.tav300s")
# msmd.open("/home/yoyisaurio/Desktop/HLTau_B6cont.calavg.tav300s")
# print(sys.argv[3])

# Apertura de datos y cabecera de datos
nombreArchivoMS = sys.argv[3]

# Datos
ms.open(nombreArchivoMS)
db = ms.getdata(['data','weight','u','v', 'data_desc_id', 'flag']);
ms.close()
d = db['data']
u = db['u']
v = db['v']
we = db['weight']
flag = db['flag']

# Metadatos
spw = db['data_desc_id']

c_constant = 2.99792458E8
lista = []
contador = 0
# limite = 1E-12
msmd.open(nombreArchivoMS)
nombreArchivoNuevoTXT = nombreArchivoMS + ".txt"
with open(nombreArchivoNuevoTXT, 'w') as archivoAEscribir:
    for indFilaDato in np.arange(d.shape[2]):
         chan_freqs = msmd.chanfreqs(spw[indFilaDato])
         for indPola in np.arange(d.shape[0]):
             filaConVisibilidades = d[indPola, :, indFilaDato]
             nuevosU = u[indFilaDato] * chan_freqs/c_constant
             nuevosV = v[indFilaDato] * chan_freqs/c_constant
             for indVisi in np.arange(d.shape[1]):
                 # if (flag[indPola, indVisi, indFilaDato] == 1 and np.abs(np.real(filaConVisibilidades[indVisi])) > limite and np.abs(np.imag(filaConVisibilidades[indVisi])) > limite and np.abs(nuevosU[indVisi]) > limite and np.abs(nuevosV[indVisi]) > limite and np.abs(we[indPola][indFilaDato]) > limite):
                 if (flag[indPola, indVisi, indFilaDato] == 0 and np.abs(np.real(filaConVisibilidades[indVisi])) != 0.0 and np.abs(np.imag(filaConVisibilidades[indVisi])) != 0.0 and np.abs(nuevosU[indVisi]) != 0.0 and np.abs(nuevosV[indVisi]) != 0.0 and np.abs(we[indPola][indFilaDato]) != 0.0):
                     contador += 1
                     archivoAEscribir.write(str(np.real(filaConVisibilidades[indVisi])) + " " + str(np.imag(filaConVisibilidades[indVisi])) + " " + str(nuevosU[indVisi]) + " " + str(nuevosV[indVisi]) + " " + str(we[indPola][indFilaDato]) + "\n")

nombreArchivoSoloCantVisiTXT = nombreArchivoMS + "cantvisi.txt"
with open(nombreArchivoSoloCantVisiTXT, 'w') as archivoAEscribir:
    archivoAEscribir.write(str(contador) + "\n");

msmd.close()
