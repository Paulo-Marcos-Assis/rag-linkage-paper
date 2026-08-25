#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_progresso.py
======================
Script para verificar o progresso da geração de notícias sintéticas.

Uso:
    python3 verificar_progresso.py
    
    # Ou monitorar continuamente (atualiza a cada 10s)
    watch -n 10 python3 verificar_progresso.py
"""

import json
import os
from datetime import datetime

STATUS_JSON = "status_execucao.json"

def formatar_tempo(minutos):
    """Formata minutos em formato legível"""
    if minutos < 1:
        return f"{int(minutos * 60)}s"
    elif minutos < 60:
        return f"{int(minutos)}min"
    else:
        horas = int(minutos / 60)
        mins = int(minutos % 60)
        return f"{horas}h{mins:02d}min"

def main():
    if not os.path.exists(STATUS_JSON):
        print("❌ Arquivo de status não encontrado.")
        print("   A geração ainda não foi iniciada ou o arquivo foi removido.")
        return
    
    try:
        with open(STATUS_JSON, "r", encoding="utf-8") as f:
            status = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao ler status: {e}")
        return
    
    # Extrair informações
    estado = status.get("status", "desconhecido")
    prog = status.get("progresso", {})
    tempo = status.get("tempo", {})
    
    atual = prog.get("atual", 0)
    total = prog.get("total", 0)
    pct = prog.get("percentual", 0)
    sucessos = prog.get("sucessos", 0)
    erros = prog.get("erros", 0)
    
    decorrido = tempo.get("decorrido_minutos", 0)
    restante = tempo.get("estimado_restante_minutos", 0)
    velocidade = tempo.get("velocidade_items_por_minuto", 0)
    ultima_atualizacao = status.get("ultima_atualizacao", "")
    
    # Exibir status
    print("=" * 70)
    print(" 📊 STATUS DA GERAÇÃO DE NOTÍCIAS SINTÉTICAS")
    print("=" * 70)
    
    # Status
    if estado == "concluido":
        print("✅ Status: CONCLUÍDO")
    elif estado == "em_execucao":
        print("🔄 Status: EM EXECUÇÃO")
    else:
        print(f"⚠️  Status: {estado.upper()}")
    
    print()
    
    # Progresso
    print("📈 PROGRESSO:")
    print(f"   {atual}/{total} pares processados ({pct:.1f}%)")
    
    # Barra de progresso visual
    barra_tamanho = 50
    preenchido = int((pct / 100) * barra_tamanho)
    barra = "█" * preenchido + "░" * (barra_tamanho - preenchido)
    print(f"   [{barra}]")
    
    print()
    print(f"   ✅ Sucessos: {sucessos}")
    print(f"   ❌ Erros:    {erros}")
    
    if erros > 0:
        taxa_erro = (erros / atual * 100) if atual > 0 else 0
        print(f"              (taxa de erro: {taxa_erro:.1f}%)")
    
    print()
    
    # Tempo
    print("⏱️  TEMPO:")
    print(f"   Decorrido:  {formatar_tempo(decorrido)}")
    
    if estado == "em_execucao" and restante > 0:
        print(f"   Restante:   {formatar_tempo(restante)} (estimado)")
        
        # Previsão de término
        if restante < 60:
            print(f"   Previsão:   Termina em ~{formatar_tempo(restante)}")
        else:
            horas_restantes = restante / 60
            print(f"   Previsão:   Termina em ~{horas_restantes:.1f} horas")
    
    print(f"   Velocidade: {velocidade:.1f} pares/min")
    
    print()
    
    # Última atualização
    if ultima_atualizacao:
        try:
            dt = datetime.fromisoformat(ultima_atualizacao)
            print(f"🕐 Última atualização: {dt.strftime('%d/%m/%Y %H:%M:%S')}")
        except:
            print(f"🕐 Última atualização: {ultima_atualizacao}")
    
    print("=" * 70)
    
    # Dicas
    if estado == "em_execucao":
        print()
        print("💡 DICAS:")
        print("   • Os arquivos são salvos a cada 10 registros")
        print("   • Você pode desligar o computador - o processo continua no servidor")
        print("   • Execute novamente este script para ver o progresso atualizado")
        print("   • Logs detalhados em: geracao_log.txt")

if __name__ == "__main__":
    main()
