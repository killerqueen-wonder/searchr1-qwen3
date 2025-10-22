import requests
import logging
from requests.exceptions import JSONDecodeError, RequestException
import argparse
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def verify_and_call_search_service(search_url: str, payload: dict, timeout: float = 5.0):
    """
    调用检索服务，并对响应进行状态码和 JSON 格式检查。
    
    :param search_url: 检索服务地址，例如 'http://127.0.0.1:8006/retrieve'
    :param payload:   要 POST 的 JSON 负载
    :param timeout:   网络请求超时时间（秒）
    :return:          解析后的 JSON（如果成功），否则抛出异常
    """
    try:
        logger.info(f"Calling search service at {search_url} with payload keys: {list(payload.keys())}")
        resp = requests.post(search_url, json=payload, timeout=timeout,proxies={"http": None, "https": None})
    except RequestException as e:
        logger.error(f"Failed to connect to search service: {e}")
        raise

    # 打印 HTTP 状态码
    if not resp.ok:
        logger.error(f"Search service returned HTTP {resp.status_code}: {resp.text[:200]!r}")
        resp.raise_for_status()

    # 尝试解析 JSON
    try:
        data = resp.json()
    except JSONDecodeError as e:
        logger.error(f"Invalid JSON from search service (status {resp.status_code}):\n{resp.text}")
        raise

    # 简单验证返回字段
    if "result" not in data:
        logger.error(f"JSON missing 'result' key: {data}")
        raise KeyError("'result' not in search response")

    logger.info(f"Search service returned {len(data['result'])} hits")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "test requests.")

    parser.add_argument('--queries', default= '抢劫罪',type=str)
    parser.add_argument('--test_url', default= "http://127.0.0.1:8006/retrieve",type=str)
    parser.add_argument('--topk', default= 3,type=int)
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    args = parser.parse_args()

    queries=args.queries
    topk=args.topk
    # 在训练脚本最开始或验证前做一次连通性检查
    # test_url = "http://127.0.0.1:8006/retrieve"
    test_url=args.test_url
    
    payload = {
            "queries": [queries],
            "topk": topk,
            "return_scores": True
        }
    # payload =    {
    #   "queries": ["What is Python?", "Tell me about neural networks."],
    #   "topk": 3,
    #   "return_scores": True
    # }
    # print(requests.post(test_url, json=payload).json())
    print(f'正在检索{queries}')
    try:
        test_response = verify_and_call_search_service(test_url, payload)
        print("Search service is up! Sample response:", test_response)
    except Exception as err:
        print("Search service verification failed:", err)
        # 根据实际情况决定：exit(1) 终止训练，或继续但不执行检索逻辑
        import sys; sys.exit(1)
