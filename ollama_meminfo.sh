#!/bin/bash
# ollama_meminfo.sh - ollamaの全モデルのメモリ情報を表示・JSON出力するスクリプト
#
# 使い方:
#   ./ollama_meminfo.sh                        # 標準出力のみ
#   ./ollama_meminfo.sh -o result.json         # JSON出力（ファイル名指定）
#   ./ollama_meminfo.sh -o                     # JSON出力（自動ファイル名）
#   OLLAMA_HOST=http://192.168.1.10:11434 ./ollama_meminfo.sh -o

set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_HOST:-http://localhost:11434}"
OUTPUT_FILE=""

# 引数処理 (-o ファイル名 または -o のみ)
if [[ "${1:-}" == "-o" ]]; then
    if [ -n "${2:-}" ] && [[ "${2:-}" != -* ]]; then
        OUTPUT_FILE="$2"
    else
        OUTPUT_FILE="ollama_meminfo_$(date +%Y%m%d_%H%M%S).json"
    fi
fi

# 色の定義
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RESET='\033[0m'

# バイト数を人間が読みやすい形式に変換
format_bytes() {
    local bytes=$1
    if [ "$bytes" -ge $((1024*1024*1024)) ]; then
        printf "%.1f GB" "$(echo "scale=1; $bytes / 1073741824" | bc)"
    elif [ "$bytes" -ge $((1024*1024)) ]; then
        printf "%.1f MB" "$(echo "scale=1; $bytes / 1048576" | bc)"
    else
        printf "%d B" "$bytes"
    fi
}

echo -e "${BOLD}=== Ollama モデル情報 ===${RESET}"
echo ""

# Ollamaが起動しているか確認
if ! curl -s --max-time 3 "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
    echo "エラー: Ollama が起動していません (${OLLAMA_BASE_URL})"
    exit 1
fi

# モデル一覧とタグ情報をまとめて取得（/api/tags は1回だけ呼ぶ）
TAGS_JSON=$(curl -s "${OLLAMA_BASE_URL}/api/tags")
MODELS=$(echo "$TAGS_JSON" | jq -r '.models[].name')

if [ -z "$MODELS" ]; then
    echo "モデルが見つかりません。"
    exit 0
fi

# JSON組み立て用配列
JSON_MODELS="[]"

# ヘッダー表示
printf "${BOLD}%-35s %-22s %-12s %-15s %-12s${RESET}\n" \
    "モデル名" "パラメータ" "量子化" "サイズ" "ファミリ"
printf '%0.s─' {1..90}
echo ""

# 各モデルの情報を取得・表示
for MODEL in $MODELS; do
    INFO=$(curl -s "${OLLAMA_BASE_URL}/api/show" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$MODEL\"}")

    QUANT=$(echo "$INFO"      | jq -r '.details.quantization_level // "不明"')
    FAMILY=$(echo "$INFO"     | jq -r '.details.family // "不明"')
    # model_info から正確なパラメータ数を取得（フォールバック: details.parameter_size）
    PARAM_COUNT=$(echo "$INFO" | jq -r '
        .model_info["general.parameter_count"]
        // .model_info["llm.parameter_count"]
        // null')
    PARAM_SIZE_LABEL=$(echo "$INFO" | jq -r '.details.parameter_size // "不明"')
    MODIFIED=$(echo "$TAGS_JSON" | jq -r --arg m "$MODEL" \
        '.models[] | select(.name==$m) | .modified_at // ""' | sed 's/T/ /;s/\..*//')
    FILE_SIZE=$(echo "$TAGS_JSON" | jq -r --arg m "$MODEL" \
        '.models[] | select(.name==$m) | .size // 0')
    DIGEST=$(echo "$TAGS_JSON" | jq -r --arg m "$MODEL" \
        '.models[] | select(.name==$m) | .digest // ""' | cut -c1-12)

    FORMATTED_SIZE=$(format_bytes "$FILE_SIZE")
    # 表示用: 正確な数値があれば併記
    if [ "$PARAM_COUNT" != "null" ] && [ -n "$PARAM_COUNT" ]; then
        PARAM_DISPLAY="${PARAM_SIZE_LABEL} (${PARAM_COUNT})"
    else
        PARAM_DISPLAY="$PARAM_SIZE_LABEL"
    fi

    printf "${CYAN}%-35s${RESET} %-22s %-12s ${GREEN}%-15s${RESET} %-12s\n" \
        "$MODEL" "$PARAM_DISPLAY" "$QUANT" "$FORMATTED_SIZE" "$FAMILY"

    # JSONオブジェクトを配列に追加
    JSON_MODELS=$(echo "$JSON_MODELS" | jq \
        --arg name "$MODEL" \
        --argjson param_count "${PARAM_COUNT:-null}" \
        --arg param_size_label "$PARAM_SIZE_LABEL" \
        --arg quant "$QUANT" \
        --arg family "$FAMILY" \
        --argjson file_size "$FILE_SIZE" \
        --arg digest "$DIGEST" \
        --arg modified "$MODIFIED" \
        '. + [{
            name: $name,
            details: {
                parameter_count: $param_count,
                parameter_size: $param_size_label,
                quantization_level: $quant,
                family: $family
            },
            disk_size_bytes: $file_size,
            digest: $digest,
            modified_at: $modified,
            loaded_memory: null
        }]')
done

echo ""
printf '%0.s─' {1..90}
echo ""

# 各モデルをロード → 計測 → アンロード
echo -e "\n${BOLD}=== メモリ計測（ロード → 計測 → アンロード）===${RESET}"
printf "${BOLD}%-35s %-15s %-15s %-10s${RESET}\n" \
    "モデル名" "総サイズ" "VRAMサイズ" "状態"
printf '%0.s─' {1..80}
echo ""

# モデルをロードする関数（/api/generate で keep_alive を設定）
load_model() {
    local model=$1
    curl -s "${OLLAMA_BASE_URL}/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"keep_alive\": 60, \"prompt\": \"\"}" \
        > /dev/null
}

# モデルをアンロードする関数（keep_alive=0 で即時解放）
unload_model() {
    local model=$1
    curl -s "${OLLAMA_BASE_URL}/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"keep_alive\": 0, \"prompt\": \"\"}" \
        > /dev/null
}

# /api/ps からモデルのメモリ情報を取得する関数
get_loaded_info() {
    local model=$1
    curl -s "${OLLAMA_BASE_URL}/api/ps" \
        | jq -c --arg m "$model" '.models[] | select(.name==$m)' 2>/dev/null || echo ""
}

for MODEL in $MODELS; do
    # すでにロード中か確認
    ALREADY_LOADED=$(curl -s "${OLLAMA_BASE_URL}/api/ps" \
        | jq -r --arg m "$MODEL" '.models[] | select(.name==$m) | .name' 2>/dev/null || echo "")

    if [ -n "$ALREADY_LOADED" ]; then
        STATUS="既存"
    else
        STATUS="ロード中"
        printf "${CYAN}%-35s${RESET} %-15s %-15s %s\r" "$MODEL" "-" "-" "ロード中..."
        load_model "$MODEL"
        STATUS="計測済"
    fi

    # /api/ps から計測
    MODEL_PS=$(get_loaded_info "$MODEL")

    if [ -n "$MODEL_PS" ]; then
        SIZE=$(echo "$MODEL_PS"      | jq -r '.size // 0')
        SIZE_VRAM=$(echo "$MODEL_PS" | jq -r '.size_vram // 0')
        FORMATTED_SIZE=$(format_bytes "$SIZE")
        FORMATTED_VRAM=$(format_bytes "$SIZE_VRAM")

        printf "${CYAN}%-35s${RESET} ${GREEN}%-15s${RESET} ${YELLOW}%-15s${RESET} %s\n" \
            "$MODEL" "$FORMATTED_SIZE" "$FORMATTED_VRAM" "$STATUS"

        # models[] の該当エントリの loaded_memory にマージ
        JSON_MODELS=$(echo "$JSON_MODELS" | jq \
            --arg name "$MODEL" \
            --argjson size "$SIZE" \
            --argjson size_vram "$SIZE_VRAM" \
            'map(if .name == $name then .loaded_memory = {
                total_bytes: $size,
                vram_bytes: $size_vram
            } else . end)')
    else
        printf "${CYAN}%-35s${RESET} %-15s %-15s %s\n" \
            "$MODEL" "取得失敗" "-" "エラー"
    fi

    # 元々ロードされていなかったものだけアンロード
    if [ "$ALREADY_LOADED" = "" ]; then
        unload_model "$MODEL"
    fi
done

echo ""

# JSON出力
if [ -n "$OUTPUT_FILE" ]; then
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    jq -n \
        --arg host "$OLLAMA_BASE_URL" \
        --arg ts "$TIMESTAMP" \
        --argjson models "$JSON_MODELS" \
        '{
            generated_at: $ts,
            ollama_host: $host,
            models: $models
        }' > "$OUTPUT_FILE"

    echo -e "${GREEN}JSON出力完了:${RESET} $OUTPUT_FILE"
    echo ""
fi