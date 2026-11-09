"""Повтор только неуспешных входов. Первичный прогон сохраняется неизменным."""
import json
import time
from datetime import datetime, timezone
from app import ROOT, Client, run, score, measured_sum, aggregate, report, WorkflowError


def main():
    initial = ROOT / 'results/live'
    target = ROOT / 'results/recovered'
    target.mkdir(parents=True, exist_ok=True)
    rows = json.loads((initial / 'runs.json').read_text())
    original = json.loads((initial / 'summary.json').read_text())
    client = Client(original['model'], retries=0, interval=5, max_tokens=4096)
    for row in rows:
        if row['success']:
            continue
        previous = {k: row[k] for k in ('error','seconds','calls','trace','total_tokens','cost_usd')}
        client.trace=[]
        start=time.perf_counter()
        output,error=None,None
        try:
            output=run(client,row['architecture'],row['text'])
        except WorkflowError as exc:
            error=str(exc)
        row.update(output=output,error=error,success=output is not None,
                   quality=score(output,row['expected']),seconds=time.perf_counter()-start,
                   calls=len(client.trace),total_tokens=measured_sum(client.trace,'total_tokens'),
                   cost_usd=measured_sum(client.trace,'cost_usd'),trace=list(client.trace),
                   previous_attempt=previous)
        (target/'runs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
        print(row['id'],row['architecture'],'OK' if output else error,flush=True)
    summary=aggregate(rows)
    original.update(metrics=summary,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        note='Повторены только ошибки первичного прогона; успешные первичные ответы сохранены. Метрики здесь относятся к последней попытке каждого входа. Полный расход включает previous_attempt.', interval=5)
    (target/'summary.json').write_text(json.dumps(original,ensure_ascii=False,indent=2))
    (target/'REPORT.md').write_text(report(rows,summary))

if __name__=='__main__': main()
