import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import app

class Fake:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []
    def call(self, stage, payload, schema):
        self.calls.append((stage, payload))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return app.validate(answer, schema)

A = {'sentiment':'negative','issue':'delivery','needs_clarification':False}
P = {'action':'investigate','reply':'Проверим обстоятельства задержки.'}

class Response:
    def __init__(self, value): self.value = value
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def read(self): return json.dumps(self.value).encode()

def envelope(content, usage=True):
    value = {'choices':[{'message':{'content':content},'finish_reason':'stop'}]}
    if usage: value['usage']={'total_tokens':100,'prompt_tokens':70,'completion_tokens':30,'cost':0.001}
    return value

class Tests(unittest.TestCase):
    def test_baseline_one_call(self):
        f=Fake([{**A,**P}]); self.assertEqual(app.run(f,'one_shot','опоздали'),{**A,**P})
        self.assertEqual(len(f.calls),1)
    def test_sequential_passes_analysis(self):
        f=Fake([A,P]); self.assertEqual(app.run(f,'sequential','опоздали'),{**A,**P})
        self.assertEqual(f.calls[1][1]['analysis'],A)
    def test_first_stage_failure_stops(self):
        f=Fake([app.WorkflowError('invalid_json')])
        with self.assertRaises(app.WorkflowError): app.run(f,'sequential','текст')
        self.assertEqual(len(f.calls),1)
    def test_second_stage_failure_no_partial_success(self):
        f=Fake([A,app.WorkflowError('network_error')])
        with self.assertRaises(app.WorkflowError): app.run(f,'sequential','текст')
        self.assertEqual(len(f.calls),2)
    def test_invalid_schema(self):
        for bad in ({**A,'needs_clarification':'false'}, {**A,'issue':'unknown'}, {**A,'extra':1}):
            with self.assertRaises(app.WorkflowError): app.validate(bad,app.ANALYSIS)
    def test_input_limit(self):
        with self.assertRaises(app.WorkflowError): app.run(Fake([]),'one_shot','x'*20001)
    def test_quality_failure_penalty(self):
        self.assertEqual(app.score(None,A),0)
        self.assertEqual(app.score(A,A),1)
    @patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-only-not-a-real-key'})
    def test_http_retry_accounting(self):
        c=app.Client('test/model')
        error=HTTPError('url',429,'rate',{},None)
        with patch('urllib.request.urlopen', side_effect=[error,Response(envelope(json.dumps(A)))]), patch('time.sleep'):
            self.assertEqual(c.call('stage',{},app.ANALYSIS),A)
        self.assertEqual(len(c.trace),2)
        self.assertIsNone(app.measured_sum(c.trace,'total_tokens'))
    @patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-only-not-a-real-key'})
    def test_auth_not_retried(self):
        c=app.Client('test/model')
        with patch('urllib.request.urlopen',side_effect=HTTPError('url',401,'auth',{},None)):
            with self.assertRaisesRegex(app.WorkflowError,'http_401'): c.call('stage',{},app.ANALYSIS)
        self.assertEqual(len(c.trace),1)
    @patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-only-not-a-real-key'})
    def test_bad_json_records_usage_without_repair(self):
        c=app.Client('test/model')
        with patch('urllib.request.urlopen',return_value=Response(envelope('bad json'))):
            with self.assertRaises(app.WorkflowError): c.call('stage',{},app.ANALYSIS)
        self.assertEqual(len(c.trace),1)
        self.assertEqual(app.measured_sum(c.trace,'total_tokens'),100)
    @patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-only-not-a-real-key'})
    def test_missing_usage_is_unknown(self):
        c=app.Client('test/model')
        with patch('urllib.request.urlopen',return_value=Response(envelope(json.dumps(A),False))):
            c.call('stage',{},app.ANALYSIS)
        self.assertIsNone(app.measured_sum(c.trace,'cost_usd'))
    @patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-only-not-a-real-key'})
    def test_free_configuration(self):
        c=app.Client('liquid/lfm-3-1.7b:free',retries=0,interval=3.2,max_tokens=4096)
        with patch('urllib.request.urlopen',return_value=Response(envelope(json.dumps(A)))) as mock:
            c.call('stage',{},app.ANALYSIS)
        body=json.loads(mock.call_args.args[0].data)
        self.assertEqual(body['model'],'liquid/lfm-3-1.7b:free')
        self.assertEqual(body['max_tokens'],4096)
        self.assertIn('НЕ массивом',body['messages'][0]['content'])
        self.assertEqual(c.retries,0)

    def test_dataset(self):
        cases=json.loads((app.ROOT/'data/test_cases.json').read_text())
        self.assertGreaterEqual(len(cases),10)
        self.assertEqual(len({x['id'] for x in cases}),len(cases))
        for case in cases: app.validate({**case['expected'],'reply':'Тест'},app.RESULT)
    def test_report_with_difference(self):
        rows=[]
        for arch,out in [('one_shot',{**A,**P}),('sequential',None)]:
            rows.append(dict(id='x',repeat=1,architecture=arch,text='test',output=out,error=None if out else 'injected',success=out is not None,quality=1 if out else 0,seconds=.1,calls=1,total_tokens=100,cost_usd=.001))
        report=app.report(rows,app.aggregate(rows))
        self.assertIn('injected',report)
        self.assertIn('-1.0000',report)

if __name__=='__main__': unittest.main()
