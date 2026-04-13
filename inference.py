import json
import os
from openai import OpenAI
import time

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url="https://intern.openxlab.org.cn/api/v1/",
)
model = "Intern-S1-Pro"


def process_line(line, client, model):
    data = {}
    try:
        data = json.loads(line)
        prompt = data["dialogs"][0]["content"] + "\n" + data["dialogs"][1]["content"]
        mat_id = data.get('id_ddm', 'unknown')

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        result = {
            "id_ddm": mat_id,
            "response": response.choices[0].message.content
        }
        time.sleep(1)
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        print(f"\n[Error] id_ddm: {data.get('id_ddm', 'unknown')} 请求失败: {e}")
        return None


test_jsonl_path = '/mnt/shared-storage-user/songdemin/user/huangjunhao/dataset_valid/mp_20_v3/test_subset_new.jsonl'
output_jsonl_path = f"/mnt/shared-storage-user/songdemin/user/huangjunhao/{model}_output_new.jsonl"


def main():
    lines = []
    with open(test_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lines.append(line)

    total_lines = len(lines)
    print(f"👉 共加载了 {total_lines} 条数据，开始单线程 API 请求...")

    success_count = 0

    try:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(lines, 1):
                print(f"正在处理 {i}/{total_lines} ...", end='\r')

                result = process_line(line, client, model)
                time.sleep(1)

                if result is not None:
                    f_out.write(result + "\n")
                    f_out.flush()
                    success_count += 1

        print(f"\n\n✅ 处理正常完成！成功生成 {success_count}/{total_lines} 条数据。")

    except KeyboardInterrupt:
        print(f"\n\n⚠️ 检测到 Ctrl+C！已安全终止程序。")
        print(f"💾 已经安全保存了前 {success_count} 条跑完的数据！")

    finally:
        print(f"📁 结果文件路径: {output_jsonl_path}")


if __name__ == '__main__':
    main()
