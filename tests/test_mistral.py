import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace as NS

import pymupdf as fitz
import mistral as app


def entry(title='A study of collective action', doi=None):
    return app.BibEntry(title=title, authors=['Jane Smith'], year=2024,
                        source='Journal', raw_entry='Smith (2024). ' + title, doi=doi)


def work(id='W1', title='A study of collective action', doi='https://doi.org/10.1234/example'):
    return dict(id='https://openalex.org/' + id, title=title, doi=doi,
                publication_year=2024, authorships=[{'author': {'display_name': 'Jane Smith'}}])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(app.os.environ, {'OPENALEX_API_KEY': 'test-openalex-key'})
        env.start()
        self.addCleanup(env.stop)

    def test_openalex_authentication(self):
        alex = app.OpenAlex()
        self.addCleanup(alex.session.close)
        self.assertEqual(alex.session.headers['Authorization'], 'Bearer test-openalex-key')
        with patch.dict(app.os.environ, {'OPENALEX_API_KEY': ' '}):
            with self.assertRaisesRegex(ValueError, 'Set OPENALEX_API_KEY'):
                app.OpenAlex()

    def test_api_error_is_actionable_and_redacted(self):
        exc = app.SDKError("Authentication failed", app.httpx.Response(401), "secret-api-key-and-document")
        message = app.extraction_error(exc)
        self.assertIn("MISTRAL_API_KEY", message)
        self.assertIn("expired", message)
        self.assertNotIn("secret-api-key-and-document", message)
        self.assertIn("network", app.extraction_error(app.httpx.ConnectError("private URL")))

    def test_surname_initials_and_rejection_reasons(self):
        for name in ['Smith J', 'Smith J.', 'Smith JJ', 'Smith, Jane', 'Jane Smith']:
            e = entry().model_copy(update={'authors': [name]})
            with patch.object(app.OpenAlex, 'get', return_value={'results': [work()]}):
                self.assertEqual(app.OpenAlex().match(e)['status'], 'matched', name)
        e = entry().model_copy(update={'year': 2023})
        with patch.object(app.OpenAlex, 'get', return_value={'results': [work()]}):
            result = app.OpenAlex().match(e)
            self.assertEqual(result['status'], 'unresolved')
            self.assertIn('Year missing or different', result['message'])

    def test_matching_and_ambiguity(self):
        alex = app.OpenAlex()
        with patch.object(alex, 'get', return_value={'results': [work()]}):
            self.assertEqual(alex.match(entry())['status'], 'matched')
        alex.cache.clear()
        with patch.object(alex, 'get', return_value={'results': [work(), work('W2')]}):
            self.assertEqual(alex.match(entry())['status'], 'unresolved')
        alex.cache.clear()
        with patch.object(alex, 'get', return_value={'results': [work(title='An unrelated paper')]}):
            self.assertEqual(alex.match(entry(doi='10.1234/example'))['status'], 'unresolved')

    def test_errors_remain_errors(self):
        with patch.object(app.OpenAlex, 'get', side_effect=ValueError('rate limit')):
            self.assertEqual(app.OpenAlex().match(entry())['status'], 'error')

    def test_boundaries(self):
        pages = [(0, 'Body\nReferences\nSmith 2024\nAppendix\nOther')]
        with patch.object(app, 'structured', return_value=app.ReferenceSection(found=True, start_line=3, end_line=3)):
            self.assertEqual(app.reference_text(None, pages), [(0, 'Smith 2024')])
        with patch.object(app, 'structured', return_value=app.ReferenceSection(found=True, start_line=0, end_line=99)):
            with self.assertRaises(ValueError):
                app.reference_text(None, pages)

    def test_storage_links_and_reruns(self):
        matched = {'status': 'matched', 'work': work(), 'candidates': []}
        unresolved = {'status': 'unresolved', 'work': None, 'candidates': []}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.sqlite3'
            source = app.Paper(title='Source')
            for _ in range(2):
                app.save_analysis(path, 'a', source, unresolved, [entry()], [matched], 'References')
            app.save_analysis(path, 'b', app.Paper(title=entry().title), matched, [entry('Third work')], [unresolved], 'References')
            db = app.connect_db(path)
            self.assertEqual(db.execute('SELECT count(*) FROM citations').fetchone()[0], 2)
            cited = db.execute("SELECT cited_id FROM citations WHERE document_sha256='a'").fetchone()[0]
            citing = db.execute("SELECT citing_id FROM citations WHERE document_sha256='b'").fetchone()[0]
            self.assertEqual(cited, citing)
            original = json.loads(db.execute("SELECT original_json FROM citations WHERE document_sha256='a'").fetchone()[0])
            self.assertIsNone(original['doi'])
            db.close()

    def test_browser_configuration_and_graph(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'DB_PATH', Path(folder) / 'test.db'), \
             patch.object(app.studio, 'Handler'), patch.object(app.studio, 'parse_pages', app.studio.parse_pages), \
             patch.object(app.studio, 'HTML', app.studio.HTML), patch.object(app.studio, 'run_extraction'), \
             patch.object(app.studio, 'MODEL'):
            app.configure_web()
            self.assertIsNone(app.studio.parse_pages('', 3))
            self.assertEqual(app.studio.parse_pages('2-3', 3), [1, 2])
            self.assertIn('View matched work', app.studio.HTML)
            handler = object.__new__(app.Handler)
            handler.path = '/api/graph'
            handler.json = Mock()
            handler.do_GET()
            handler.json.assert_called_once_with(200, {'papers': [], 'citations': []})

    def test_identity_expands_only_when_needed(self):
        reader = Mock(local_texts=['cover', 'paper', 'body'])
        reader.read.side_effect = [[(0, 'cover')], [(0, 'cover'), (1, 'paper')]]
        with patch.object(app, 'structured', side_effect=[app.Paper(title=''), app.Paper(title='Real paper')]):
            result = app.identify_paper(None, reader, lambda _: None)
        self.assertEqual(result.title, 'Real paper')
        self.assertEqual([c.args[0] for c in reader.read.call_args_list], [[0], [0, 1]])

    def test_local_reference_detection(self):
        texts = ['Title', 'Contents\nReferences ........ 8', 'Body', '7 Literaturverzeichnis\nSmith', 'More references']
        self.assertEqual(app.likely_reference_pages(texts), [3, 4])
        self.assertEqual(app.likely_reference_pages(['scanned', '']), [])

    def test_candidate_failure_falls_back(self):
        for candidate_result in [app.NoReferences('none'), [(2, 'References')]]:
            with self.subTest(candidate_result=candidate_result), tempfile.TemporaryDirectory() as folder, \
                 patch.object(app, 'DB_PATH', Path(folder) / 'test.db'), \
                 patch.dict(app.os.environ, {'MISTRAL_API_KEY': 'test'}), \
                 patch.object(app, 'Mistral') as factory, patch.object(app, 'PDFReader') as reader_factory, \
                 patch.object(app, 'identify_paper', return_value=app.Paper(title='Source')), \
                 patch.object(app, 'reference_text', side_effect=[candidate_result, [(1, 'Actual refs')]]) as locate, \
                 patch.object(app, 'extract_bibliography', side_effect=([[], [entry()]] if isinstance(candidate_result, list) else [[entry()]])), \
                 patch.object(app.OpenAlex, 'match', return_value={'status': 'unresolved', 'work': None, 'candidates': []}):
                reader = reader_factory.return_value
                reader.local_texts = ['Title', 'Body', 'References']
                reader.read.side_effect = [[(2, 'References')], [(0, 'Title'), (1, 'Actual refs'), (2, 'References')]]
                result = app.analyze(b'PDF')
                self.assertEqual(result['reference_pages'], [2])
                self.assertEqual(locate.call_count, 2)
                self.assertEqual(reader.read.call_args_list[1].args[0], [0, 1, 2])

    def test_bulk_doi_skips_search_and_preserves_order(self):
        alex = app.OpenAlex()
        items = [entry(doi='10.1234/example'), entry('Unmatched title')]
        def lookup(path, params):
            if 'filter' in params:
                self.assertIn('doi:10.1234/example', params['filter'])
                return {'results': [work()]}
            self.assertEqual(params['search'], 'Unmatched title')
            return {'results': []}
        with patch.object(app.OpenAlex, 'get', side_effect=lookup) as get:
            results = alex.match_many(items, lambda _: None)
        self.assertEqual([r['status'] for r in results], ['matched', 'unresolved'])
        self.assertEqual(get.call_count, 2)
        alex.session.close()

    def test_cache_survives_clients_and_expires(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'DB_PATH', Path(folder) / 'test.db'), \
             patch.object(app.OPENALEX_LIMIT, 'wait'):
            response = Mock(status_code=200, ok=True)
            response.json.return_value = {'results': [work()]}
            for _ in range(2):
                alex = app.OpenAlex()
                with patch.object(alex.session, 'get', return_value=response) as get:
                    self.assertEqual(alex.get('/works', {'search': 'paper'}), {'results': [work()]})
                    self.assertEqual(get.call_count, 1 if _ == 0 else 0)
                alex.session.close()
            db = app.connect_db(app.DB_PATH)
            with db:
                db.execute('UPDATE api_cache SET expires=0')
            db.close()
            with patch.object(alex.session, 'get', return_value=response) as get:
                alex.get('/works', {'search': 'paper'})
                get.assert_called_once()

    def test_ocr_only_uploads_missing_pages(self):
        with fitz.open() as pdf:
            for i in range(3):
                pdf.new_page().insert_text((72, 72), f'page {i}')
            data = pdf.tobytes()
        uploaded = []
        def ocr(**kwargs):
            payload = kwargs['document']['document_url'].split(',', 1)[1]
            with fitz.open(stream=app.base64.b64decode(payload), filetype='pdf') as subset:
                text = [page.get_text().strip() for page in subset]
            uploaded.append(text)
            return NS(pages=[NS(index=i, markdown=t) for i, t in enumerate(text)])
        client = Mock()
        client.ocr.process.side_effect = ocr
        reader = app.PDFReader(client, data, {}, lambda _: None)
        self.assertEqual(reader.read([0]), [(0, 'page 0')])
        self.assertEqual(reader.read([0, 1]), [(0, 'page 0'), (1, 'page 1')])
        reader.read([0, 1, 2], references=True)
        self.assertEqual(uploaded, [['page 0'], ['page 1'], ['page 2']])

    def test_rate_limiter_spaces_requests(self):
        limiter = app.RequestLimiter(.2)
        with patch.object(app.time, 'monotonic', return_value=10), patch.object(app.time, 'sleep') as sleep:
            limiter.wait()
            sleep.assert_not_called()
            limiter.wait()
            self.assertAlmostEqual(sleep.call_args.args[0], .2)

    def test_review_persists_history_and_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.db'
            candidate = work()
            match = {'status': 'unresolved', 'work': None, 'candidates': [{'work': candidate}]}
            source_match = {'status': 'unresolved', 'work': None, 'candidates': [{'work': work('W9', 'Source')}]}
            app.save_analysis(path, 'doc', app.Paper(title='Source'), source_match, [entry()], [match], 'refs')
            db = app.connect_db(path)
            with db:
                app.import_legacy_runs(db)
            legacy = app.load_run(db, 'legacy-doc')
            self.assertEqual(legacy['counts']['unresolved'], 1)
            with db:
                chosen = app.choose_match(db, 'legacy-doc', 0, candidate['id'])
            self.assertEqual(chosen['counts']['confirmed'], 1)
            self.assertIn('10.1234/example', chosen['bibtex'])
            self.assertIsNone(chosen['original_entries'][0]['doi'])
            self.assertEqual(json.loads(db.execute('SELECT match_json FROM citations').fetchone()[0])['decision'], 'manual')
            with self.assertRaises(ValueError):
                app.choose_match(db, 'legacy-doc', 0, 'https://openalex.org/W999')
            newer = json.loads(json.dumps(legacy))
            with db:
                app.record_run(db, 'doc', newer, 'new-run', created=123)
                app.choose_match(db, 'legacy-doc', 0, None)
            # Reviewing an older run must not rewrite the latest graph.
            self.assertEqual(json.loads(db.execute('SELECT match_json FROM citations').fetchone()[0])['decision'], 'manual')
            self.assertEqual(app.load_run(db, 'legacy-doc')['counts']['identified'], 0)
            db.close()
            db = app.connect_db(path)
            self.assertEqual(app.load_run(db, 'new-run')['counts']['total'], 1)
            db.close()

    def test_bibtex_uses_publication_metadata(self):
        w = work()
        w.update(type='article', primary_location={'source': {'type': 'journal', 'display_name': 'Science & Society'}},
                 biblio={'volume': '12', 'issue': '3', 'first_page': '20', 'last_page': '29'})
        match = {'work': w}
        e = app.canonical(entry(), match)
        bib = app.export_records([e], [match])[0]
        for field in ['@article', r'journal = {Science \& Society}', 'volume = {12}', 'number = {3}', 'pages = {20--29}', 'doi = {10.1234/example}']:
            self.assertIn(field, bib)
        self.assertNotIn('howpublished', bib)
        w.update(type='book-chapter', volume={'display_name': 'Collected Essays'}, primary_location={'source': {'type': 'book series', 'display_name': 'Series name'}})
        chapter = app.export_records([app.canonical(entry(), match)], [match])[0]
        self.assertIn('@incollection', chapter)
        self.assertIn('booktitle = {Collected Essays}', chapter)
        self.assertNotIn('booktitle = {Series name}', chapter)
        w.update(type='book')
        book = app.export_records([app.canonical(entry(), match)], [match])[0]
        self.assertIn('@book', book)
        self.assertIn('publisher = {Journal}', book)
        unknown = app.export_records([entry()], [{'work': None}])[0]
        self.assertIn('@misc', unknown)

    def test_full_pipeline_mocked_services(self):
        pdf = fitz.open()
        for _ in range(3):
            pdf.new_page()
        data = pdf.tobytes()
        pdf.close()
        client = Mock()
        client.ocr.process.side_effect = [NS(pages=[NS(index=0, markdown='page 0')]), NS(pages=[NS(index=0, markdown='page 2')])]
        unresolved = {'status': 'unresolved', 'work': None, 'candidates': []}
        with tempfile.TemporaryDirectory() as folder, patch.dict(app.os.environ, {'MISTRAL_API_KEY': 'test'}), \
             patch.object(app, 'DB_PATH', Path(folder) / 'test.db'), \
             patch.object(app, 'Mistral', return_value=client), \
             patch.object(app, 'structured', side_effect=[app.Paper(title='Source'), app.Bibliography(entries=[entry()])]) as extract, \
             patch.object(app.OpenAlex, 'match', return_value=unresolved):
            client.__exit__ = Mock()
            result = app.analyze(data, pages=[2])
            self.assertEqual(extract.call_args_list[0].args[3], 'page 0')
            self.assertEqual(extract.call_args_list[1].args[3], 'page 2')
            self.assertEqual(result['reference_pages'], [3])
            self.assertIn('@misc', result['bibtex'])


if __name__ == '__main__':
    unittest.main()
