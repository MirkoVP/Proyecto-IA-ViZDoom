# Proyecto IA 2025 ViZDoom

## Como hacerlo funcionar

1. Clonar el repositorio en el subsistema linux (yo use wsl)
```
git clone https://github.com/MirkoVP/Proyecto-IA-ViZDoom.git
```
2. Crear un ambiente
```
python3 -m venv venv
```
3. Activar el ambiente 
```
source venv/bin/activate
```
4. Instalar librerias necesarias con "pip install * nombre de la libreria *" o requirements.txt
   - Librerias: vizdoom, stable-baselines3[extra], numpy, matplotlib y scikit-image
   - Deberia ser posible usar "pip install -r requirements.txt", actualice recien el archivo con solo lo necesario, si lo prueban avisenme si funciona
```
pip install vizdoom stable-baselines3[extra] numpy matplotlib scikit-image

o

pip install -r requirements.txt
```
5. Crear una nueva carpeta dentro de la carpeta del repositorio (yo la llame githubVizDoom por lo tanto asi aparece en el codigo)
```
mkdir githubVizDoom
cd githubVizDoom
```
6. Dentro de esa carpeta copiar el repositorio de vizdoom 
```
git clone https://github.com/Farama-Foundation/ViZDoom.git
```
8. Abrir vscode desde la consola
```
code .
```
9. Instalar la extension wsl y reiniciar vscode
10. Setear el interprete, apreten la seach bar de arriba de vscode o "crtl+shift+P" y escriban 
```
>Python: Select Interpreter
```
y elijan el que sea como venv/bin/python. CREO que con la extension de wsl esto lo hace automaticamente al abrir vscode desde la terminal de ubuntu con el ambiente activado
11. Ahora deberia funcionar test1.py, yo uso la extension autorunner para correr codigo
12. (EXTRA) Para desactivar el ambiente pueden usar
```
deactivate
```

Con todo eso deberia funcionar el archivo test.py del repositorio

ps. Yo use python 3.12.3

ᓚᘏᗢ
