"""발급받은 API 키와 엔드포인트를 최소 호출로 검증한다.

    python scripts/check_keys.py

각 항목마다 OK / FAIL / SKIP(키 미입력) 을 출력한다.
특히 [KMA-3] 중기예보 과거 조회는 이 프로젝트 설계의 전제이므로 반드시 OK 여야 한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ROOT / "configs" / "secrets.yaml"
KMA = "https://apihub.kma.go.kr/api/typ01/url"
TIMEOUT = 30

PASS, FAIL, SKIP = "\033[32mOK  \033[0m", "\033[31mFAIL\033[0m", "\033[33mSKIP\033[0m"


def report(tag: str, status: str, detail: str = "") -> bool:
    print(f"  [{status}] {tag}" + (f" — {detail}" if detail else ""))
    return status == PASS


def preview(text: str, n: int = 160) -> str:
    """응답 앞부분을 한 줄로."""
    return " ".join(text.split())[:n]


def kma_ok(text: str) -> tuple[bool, str]:
    """API 허브 응답 판정.

    정상: `#START7777` ~ `#7777END` 로 감싼 텍스트, 그 사이가 데이터 행.
    실패: JSON 에러 바디 (403 활용신청 필요, 인증 실패 등) — 상태코드가 200인 경우도 있다.
    """
    t = text.strip()
    if t.startswith("{"):
        try:
            return False, json.loads(t)["result"]["message"]
        except Exception:
            return False, preview(t)
    rows = [ln for ln in t.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        return False, f"데이터 0행 — {preview(t, 80)}"
    return True, f"{len(rows)}행 수신"


def check_kma(key: str) -> None:
    print("\n■ 기상청 API 허브")
    if not key:
        report("전체", SKIP, "kma_apihub.auth_key 비어 있음")
        return

    # 1) 지상관측 일자료 — 백필의 기준선
    r = requests.get(f"{KMA}/kma_sfcdd3.php", timeout=TIMEOUT, params={
        "tm1": "20230101", "tm2": "20230107", "stn": "108", "disp": "1", "help": "0", "authKey": key})
    ok, msg = kma_ok(r.text)
    report("[KMA-1] 지상관측 일자료 (서울 108, 2023-01-01~07)", PASS if ok else FAIL, msg)

    # 2) 특보 발표 이력
    r = requests.get(f"{KMA}/wrn_met_data.php", timeout=TIMEOUT, params={
        "tmfc1": "20230101", "tmfc2": "20230131", "disp": "1", "help": "0", "authKey": key})
    ok, msg = kma_ok(r.text)
    report("[KMA-2] 기상특보 이력 (2023-01)", PASS if ok else FAIL, msg)

    # 3) 중기 기온예보 과거 발표분 — ★ 설계 전제. 실패 시 모델 입력 전략을 바꿔야 한다
    r = requests.get(f"{KMA}/fct_afs_wc.php", timeout=TIMEOUT, params={
        "tmfc1": "2023010106", "tmfc2": "2023010218", "reg": "11B10101",
        "disp": "1", "help": "0", "authKey": key})
    ok, msg = kma_ok(r.text)
    report("[KMA-3] ★ 중기 기온예보 과거 발표분 (2023-01)", PASS if ok else FAIL, msg)
    if not ok and "활용신청" not in msg:
        # 403(권한)은 설계 문제가 아니다. 데이터가 정말 안 올 때만 설계 재검토 대상.
        print("         └ 과거 발표분 조회 불가 → 예보 피처는 오늘부터 축적으로 전환,")
        print("           3년 백필은 관측(KMA-1)만 사용하도록 설계 변경 필요")

    # 4) 중기 육상예보
    r = requests.get(f"{KMA}/fct_afs_wl.php", timeout=TIMEOUT, params={
        "tmfc1": "2023010106", "tmfc2": "2023010218", "reg": "11B00000",
        "disp": "1", "help": "0", "authKey": key})
    ok, msg = kma_ok(r.text)
    report("[KMA-4] 중기 육상예보 과거 발표분", PASS if ok else FAIL, msg)


def check_naver(cid: str, secret: str, hub_id: str, hub_key: str) -> None:
    """데이터랩은 NAVER API HUB로 이관됐다(2026-07-31).

    기존 개발자센터 방식은 유예 대상자에 한해 2027-06-30까지 동작한다.
    8주 프로젝트는 유예 기간 안에 끝나므로, 기존 방식이 살아있으면 그대로 쓴다.
    """
    span = {"startDate": "2026-07-01", "endDate": "2026-07-07", "timeUnit": "date"}
    shopping = {**span, "category": [{"name": "패션의류", "param": ["50000000"]}]}
    trend = {**span, "keywordGroups": [{"groupName": "패딩", "keywords": ["패딩"]}]}

    print("\n■ 네이버 데이터랩 — 기존 방식 (개발자센터, 유예 2027-06-30)")
    legacy_ok = False
    if not (cid and secret):
        report("전체", SKIP, "naver.client_id / client_secret 비어 있음")
    else:
        hdr = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": secret,
               "Content-Type": "application/json"}
        for tag, path, body, hint in [
            ("[NAVER-1] 쇼핑인사이트 카테고리", "/v1/datalab/shopping/categories", shopping,
             "앱에 '데이터랩(쇼핑인사이트)' API가 추가돼 있는지 확인"),
            ("[NAVER-2] 검색어트렌드", "/v1/datalab/search", trend,
             "'데이터랩(검색어트렌드)'는 쇼핑인사이트와 별도 권한"),
        ]:
            r = requests.post(f"https://openapi.naver.com{path}", headers=hdr,
                              timeout=TIMEOUT, json=body)
            if r.status_code == 200:
                legacy_ok |= report(tag, PASS, f"{len(r.json()['results'][0]['data'])}일치 수신")
            else:
                report(tag, FAIL, f"{r.status_code} {preview(r.text)}")
                print(f"         └ {hint}")

    print("\n■ 네이버 데이터랩 — API HUB (네이버클라우드, 신규 표준)")
    # NCP 키는 과금 계정에 연결된다. 한시적 무료이지만 불필요한 호출을 만들지 않는다.
    # 기존 방식이 동작하면 HUB는 건드리지 않고, 필요할 때만 --hub 로 명시 호출한다.
    if legacy_ok and "--hub" not in sys.argv:
        report("전체", SKIP, "기존 방식이 동작하므로 호출하지 않음 (강제: --hub)")
    elif not (hub_id and hub_key):
        report("전체", SKIP, "naver_api_hub 키 비어 있음"
               + ("" if legacy_ok else " — 기존 방식도 실패했다면 HUB 발급 필요"))
    else:
        print("  ※ NCP 과금 계정 호출 — 검증 목적 2회만 수행")
        hdr = {"X-NCP-APIGW-API-KEY-ID": hub_id, "X-NCP-APIGW-API-KEY": hub_key,
               "Content-Type": "application/json"}
        base = "https://naverapihub.apigw.ntruss.com"
        # 경로는 공개 문서에 없어 추정값이다. 404면 콘솔의 API 문서에서 실제 경로를 확인할 것.
        for tag, path, body in [
            ("[HUB-1] 쇼핑인사이트 카테고리", "/datalab/v1/shopping/categories", shopping),
            ("[HUB-2] 검색어트렌드", "/datalab/v1/search", trend),
        ]:
            try:
                r = requests.post(base + path, headers=hdr, timeout=TIMEOUT, json=body)
            except requests.RequestException as e:
                report(tag, FAIL, str(e)[:120])
                continue
            if r.status_code == 200:
                report(tag, PASS, f"{len(r.json()['results'][0]['data'])}일치 수신")
            elif r.status_code == 404:
                report(tag, FAIL, f"404 — 경로 추정 실패({path})")
                print("         └ NCP 콘솔 > NAVER API HUB > API 문서에서 실제 경로 확인 후 수정")
            else:
                report(tag, FAIL, f"{r.status_code} {preview(r.text)}")

    if legacy_ok:
        print("\n  → 기존 방식이 동작한다. 유예 기한(2027-06-30)이 프로젝트 종료보다 뒤이므로")
        print("    이번 프로젝트는 개발자센터 방식으로 진행해도 된다.")

    print("\n■ 네이버 쇼핑 검색 API")
    report("[NAVER-3] /v1/search/shop.json", SKIP,
           "2026-07-31 완전 종료, 공식 대체 없음 → 데이터셋 T6 폐기")


def check_datagokr(key: str) -> None:
    print("\n■ 공공데이터포털")
    if not key:
        report("공휴일", SKIP, "data_go_kr.service_key 비어 있음")
        return
    r = requests.get(
        "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo",
        timeout=TIMEOUT,
        params={"serviceKey": key, "solYear": "2025", "_type": "json", "numOfRows": 30})
    # 인증 실패는 resultCode 로 판정한다. 본문에 "NORMAL SERVICE" 가 들어가므로
    # 문자열 'SERVICE' 유무로 판정하면 정상 응답을 실패로 오판한다.
    ok, detail = False, preview(r.text)
    if r.status_code == 200:
        try:
            body = json.loads(r.text)["response"]
            code = body["header"]["resultCode"]
            ok = code == "00"
            detail = f"{len(body['body']['items']['item'])}건 수신" if ok else f"resultCode={code}"
        except Exception:
            ok = False  # XML 에러 바디 (SERVICE_KEY_IS_NOT_REGISTERED_ERROR 등)
    report("[GOV-1] 공휴일 (특일 정보 2025)", PASS if ok else FAIL, detail)
    if not ok:
        print("         └ Decoding 키를 넣었는지 확인 (Encoding 키는 이중 인코딩되어 실패)")


def main() -> int:
    if not SECRETS.exists():
        print(f"{SECRETS} 가 없습니다. secrets.example.yaml 을 복사하세요.")
        return 1

    s = yaml.safe_load(SECRETS.read_text(encoding="utf-8")) or {}
    get = lambda *ks: (s.get(ks[0]) or {}).get(ks[1]) or ""

    print(f"검증 대상: {SECRETS}")
    check_kma(get("kma_apihub", "auth_key"))
    check_naver(get("naver", "client_id"), get("naver", "client_secret"),
                get("naver_api_hub", "access_key_id"), get("naver_api_hub", "secret_key"))
    check_datagokr(get("data_go_kr", "service_key"))
    print("\n※ [KMA-3] 이 FAIL 이면 데이터 수집 전에 설계를 먼저 조정해야 한다.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
