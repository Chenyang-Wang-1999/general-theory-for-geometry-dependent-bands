'''
author:        Wang Chenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-06-13
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''
import re
import sys
from collections import OrderedDict

def find_unique_regex_matches(file_path, pattern):
    """
    在文件中查找所有匹配正则表达式的唯一结果
    :param file_path: 文件路径
    :param pattern: 正则表达式字符串
    :return: 无重复的匹配结果列表
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 编译正则表达式
        regex = re.compile(pattern)
        
        # 使用有序字典保持顺序并去重
        unique_matches = OrderedDict()
        
        # 查找所有匹配项
        for match in regex.finditer(content):
            matched_text = match.group(0)
            unique_matches[matched_text] = True  # 值无关紧要，只为去重
        
        return list(unique_matches.keys())
    
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        return []
    except re.error:
        print(f"错误: 无效的正则表达式 '{pattern}'")
        return []
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python regex_finder.py <文件路径> <正则表达式>")
        print("示例: python regex_finder.py document.txt '[A-Z]{2}'")
        sys.exit(1)
    
    file_path = sys.argv[1]
    regex_pattern = sys.argv[2]
    
    results = find_unique_regex_matches(file_path, regex_pattern)
    
    if results:
        print(f"找到 {len(results)} 个唯一匹配项:")
        for i, match in enumerate(results, 1):
            print(f"{i}. {match}")
    else:
        print("未找到匹配项")