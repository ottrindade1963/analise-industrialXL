"""
Script auxiliar para visualizar imagens geradas pelo pipeline.
Uso: python visualizar_imagens.py [diretorio]
"""
import os
import sys
from pathlib import Path

def listar_imagens(diretorio):
    """Lista todas as imagens PNG em um diretório."""
    if not os.path.exists(diretorio):
        print(f"❌ Diretório não encontrado: {diretorio}")
        return []
    
    imagens = []
    for root, dirs, files in os.walk(diretorio):
        for f in sorted(files):
            if f.lower().endswith('.png'):
                imagens.append(os.path.join(root, f))
    
    return imagens

def main():
    if len(sys.argv) > 1:
        diretorio = sys.argv[1]
    else:
        diretorio = 'analise_estrategias'
    
    # Resolver caminho absoluto
    if not os.path.isabs(diretorio):
        diretorio = os.path.join(os.getcwd(), diretorio)
    
    imagens = listar_imagens(diretorio)
    
    if not imagens:
        print(f"❌ Nenhuma imagem PNG encontrada em: {diretorio}")
        return
    
    print(f"\n✓ {len(imagens)} imagens encontradas em {diretorio}:\n")
    for i, img in enumerate(imagens, 1):
        nome = os.path.basename(img)
        tamanho = os.path.getsize(img) / 1024  # KB
        print(f"  {i}. {nome} ({tamanho:.1f} KB)")
        print(f"     Caminho: {img}")

if __name__ == '__main__':
    main()
