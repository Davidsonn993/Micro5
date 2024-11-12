import serial
import csv

# Configura el puerto serial
PuertoSerial = serial.Serial(port='COM8', baudrate=115200, timeout=1)

# Abrir el archivo CSV para escribir
with open("tensiones.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    
    try:
        while True:
            # Leer una línea del puerto serial
            valor = PuertoSerial.readline().decode('utf-8').strip()
            
            # Verificar si la línea está vacía
            if not valor:
                print("No se recibió ningún dato o la línea está vacía.")
                continue

            # Mostrar los datos recibidos para verificar
            print(f"Datos recibidos: {valor}")
            
            # Separar la clave (V1, V2, etc.) del valor (-20.38, etc.)
            datos_separados = valor.split(',')
            
            # Verificar si los datos están en el formato esperado
            if len(datos_separados) == 6:
                writer.writerow(datos_separados)
            else:
                print(f"Formato inesperado: {valor}")
    
    except KeyboardInterrupt:
        print("Programa interrumpido.")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        PuertoSerial.close()
        print("Puerto serial cerrado.")


