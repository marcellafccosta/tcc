#!/bin/bash
# Script para ativar o ambiente virtual e rodar testes

cd "$(dirname "$0")"

# Ativa o ambiente virtual
source venv/bin/activate

echo "✓ Ambiente virtual ativado"
echo "✓ Python: $(which python3)"
echo ""
echo "Para rodar o teste:"
echo "  python3 testar_3_videos.py"
echo ""
echo "Para desativar:"
echo "  deactivate"
echo ""

# Executa o shell com o venv ativado
exec $SHELL
