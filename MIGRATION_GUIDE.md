# Guía de Migración a Dependencias Modernas

## Cambios Realizados

### 1. Actualización de Dependencias
- **Python**: 3.7.3 → 3.10+ (o superior)
- **TensorFlow**: 1.14.0 → 2.13.0+
- **Keras**: 2.2.5 (separado) → Integrado en TensorFlow 2.x
- **NumPy**: Compatible con la nueva versión (1.24.0+)
- **SciPy**: Compatible con la nueva versión (1.10.0+)

### 2. Cambios de Imports
Todos los imports de `keras` han sido reemplazados con `tensorflow.keras`:

```python
# ANTES (Keras separado)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping

# AHORA (TensorFlow 2.x)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
```

### 3. Archivos Actualizados
- `models/PGNN.py` - Migrado a TensorFlow 2.x
- `hybrid/hpd.py` - Migrado a TensorFlow 2.x
- `hybrid/res_nn.py` - Migrado a TensorFlow 2.x

### 4. Cambios de Comportamiento Importantes en TensorFlow 2.x

#### 4.1 Eager Execution (Habilitado por defecto)
TensorFlow 2.x ejecuta operaciones inmediatamente. El código de física debe adaptarse.

#### 4.2 Composición de Modelos
Las pérdidas personalizadas y las regularizaciones físicas pueden necesitar ajustes.

#### 4.3 Optimizadores
Se mantienen los optimizadores existentes, pero con comportamiento mejorado:
- `Adam`, `Adadelta`, `RMSprop`, `SGD` - Compatible
- `Nadam` - Compatible
- Parámetros por defecto pueden haber cambiado ligeramente

#### 4.4 Backend de Keras
`K.` (backend) funciona principalmente igual, pero algunos comportamientos pueden diferir.

### 5. Compatibilidades Verificadas
- ✅ Carga/Guardado de modelos `.h5` - Compatible
- ✅ Matrices de datos `.mat` (scipy) - Compatible
- ✅ Callbacks: `EarlyStopping`, `TerminateOnNaN` - Compatible
- ✅ Optimizadores especificados - Compatible

### 6. Pasos para Usar la Nueva Rama

#### 6.1 Limpiar Ambiente Antiguo
```bash
# Opcionalmente, remover el antiguo ambiente
rm -rf .venv37
```

#### 6.2 Crear Nuevo Ambiente Virtual
```bash
python3.10 -m venv .venv
# o
python3.11 -m venv .venv
```

#### 6.3 Activar y Instalar Dependencias
```bash
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

#### 6.4 Ejecutar Scripts
```bash
# Entrenar modelo PGNN
cd models
python PGNN.py

# O usar versión con argumentos
cd ../hybrid
python hpd.py --dataset mille_lacs --optimizer_val 2 --lamda 100.0
```

### 7. Notas Importantes

#### 7.1 Cambios de Sintaxis Menores
- Se removió `from __future__ import print_function` (Python 3 solo)
- Los cambios son principalmente de import, la lógica del modelo se mantiene

#### 7.2 Pérdida Física Personalizada
La función `combined_loss()` usa `K.constants()` y operaciones tensoriales.
En TensorFlow 2.x con eager execution, esto funciona correctamente.

#### 7.3 Cuadernos Jupyter
Los cuadernos (`.ipynb`) pueden necesitar:
- Reinstalar el kernel
- Actualizar cualquier import de keras

### 8. Testing Recomendado
1. Entrenar un modelo pequeño primero
2. Comparar resultados con versión antigua
3. Validar pérdidas (RMSE y física)
4. Verificar guardado/carga de modelos

### 9. Rollback (Si es Necesario)
```bash
git checkout master
source .venv37/bin/activate  # Usar ambiente antiguo
```

## Compatibilidad Hacia Atrás
Los modelos guardados en TensorFlow 1.x pueden no ser directamente compatibles.
Se recomienda reentrenar modelos con la nueva versión.

## Referencias
- [TensorFlow 2.x Migration Guide](https://www.tensorflow.org/guide/migrate)
- [Keras API en TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras)
