#!/usr/bin/env python3
"""Validador para la API de Gemini."""

import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    from scripts.core.utils import load_settings
    # Cargar variables de entorno
    load_dotenv(root / ".env")
except ImportError as e:
    print(f"❌ Error importando dependencias: {e}")
    print("Ejecuta: pip install google-generativeai python-dotenv")
    sys.exit(1)

def validate_gemini_api():
    """Valida que las APIs de Gemini funcionen correctamente."""
    # Cargar configuración
    cfg = load_settings(root)
    models = cfg.get("analysis", {}).get("models", [cfg.get("analysis", {}).get("default_model", "gemini-1.5-flash")])
    if isinstance(models, str):
        models = [models]

    # Verificar APIs
    api_keys = []
    for i in range(1, 4):
        key = os.getenv(f"GEMINI_API_KEY_{i}", "")
        if key and key != "your_gemini_api_key_here":
            api_keys.append(key)

    if not api_keys:
        print("❌ Ninguna GEMINI_API_KEY configurada en .env")
        print("Configura GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.")
        return False

    print(f"🔑 Encontradas {len(api_keys)} APIs configuradas")
    print(f"🤖 Modelos a probar: {', '.join(models)}")

    working_combinations = []

    for i, api_key in enumerate(api_keys, 1):
        print(f"\n🔄 Probando API {i}...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            for model in models:
                try:
                    # Probar una llamada simple
                    test_model = genai.GenerativeModel(model)
                    response = test_model.generate_content("Hola, ¿funciona la API?")
                    if response.text and len(response.text.strip()) > 0:
                        working_combinations.append((i, model))
                        print(f"✅ API {i} + {model}: OK")
                    else:
                        print(f"⚠️ API {i} + {model}: Respuesta vacía")
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        print(f"⏳ API {i} + {model}: Cuota excedida (reintentar en unos minutos)")
                    else:
                        print(f"❌ API {i} + {model}: Error - {e}")

        except Exception as e:
            print(f"❌ Error configurando API {i}: {e}")

    if working_combinations:
        print(f"\n🎉 ¡{len(working_combinations)} combinaciones funcionan!")
        for api_num, model in working_combinations:
            print(f"  - API {api_num} con {model}")
        print("\n💡 El sistema usará automáticamente las APIs que funcionen.")
        return True
    else:
        print("\n❌ Ninguna combinación API + modelo funciona.")
        print("Verifica tus claves API y cuotas en https://aistudio.google.com/app/apikey")
        return False

if __name__ == "__main__":
    success = validate_gemini_api()
    sys.exit(0 if success else 1)