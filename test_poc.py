import unittest
from poc import (group_documents, encode_labels, decode_spans, class_weights, match_iou,
                 continuation_multipliers, split_dataset, HEAD_A, boundary_mask_and_weights,
                 postprocess_spans_v3, predict_relations_v3)


class PocTests(unittest.TestCase):
    def test_transitive_groups(self):
        docs=[{'id':'a','hash':'x','cnj':['1']},{'id':'b','hash':'x','cnj':['2']},{'id':'c','hash':'y','cnj':['2']},{'id':'d','hash':'z','cnj':['3']}]
        groups=group_documents(docs)
        self.assertEqual(groups['a'],groups['c'])
        self.assertNotEqual(groups['a'],groups['d'])

    def test_nested_and_unknown(self):
        offsets=[(0,0),(0,4),(5,8),(9,13),(0,0)]
        spans=[{'label':'Encaminhamento','begin':0,'end':13},{'label':'Prazo','begin':5,'end':8},{'label':'desconhecido','begin':9,'end':13}]
        a,b=encode_labels(offsets,spans)
        self.assertEqual(a,[-100,3,4,-100,-100])
        self.assertEqual(b,[-100,0,1,-100,-100])

    def test_roundtrip_bio(self):
        self.assertEqual(decode_spans([(0,3),(4,8),(9,12)],[1,2,0],['O','B-Pronunciamento','I-Pronunciamento']), [('Pronunciamento',0,8)])

    def test_smoothed_weights_ignore_mask_and_unsupported_classes(self):
        w=class_weights([0,0,0,0,1,-100],3,.5)
        self.assertAlmostEqual(w[1]/w[0],2.)
        self.assertEqual(w[2],0.)

    def test_iou_matching_is_one_to_one_and_respects_document_and_family(self):
        gold={('a','Prazo',0,10),('b','Prazo',0,10)}
        pred={('a','Prazo',0,8),('a','Prazo',1,9),('b','Pronunciamento',0,10)}
        self.assertEqual(match_iou(gold,pred,.5),1)

    def test_continuation_weight_only_applies_to_inside_tokens(self):
        self.assertEqual(continuation_multipliers([-100,0,1,2,3,4],HEAD_A,2),[1,1,1,2,1,2])

    def test_fold_changes_dev_but_keeps_original_test_frozen(self):
        docs=[dict(id=str(i),hash=str(i),cnj=[str(i)],curated=True,versions=[dict(spans=[])]) for i in range(6)]
        first,_=split_dataset(docs,fold=0);second,_=split_dataset(docs,fold=1)
        self.assertEqual({k for k,v in first.items() if v=='test'},{k for k,v in second.items() if v=='test'})
        self.assertNotEqual({k for k,v in first.items() if v=='dev'},{k for k,v in second.items() if v=='dev'})

    def test_boundary_mask_and_soft_weights(self):
        a_ann = [-100, 1, 2, 2]
        a_gold = [0, 0, 1, 2]
        am, bm, wa, wb = boundary_mask_and_weights(a_ann, [-100]*4, a_gold, [0]*4, 'border_mask')
        self.assertEqual(am, [-100, -100, -100, 2])
        asoft, _, wasoft, _ = boundary_mask_and_weights(a_ann, [-100]*4, a_gold, [0]*4, 'border_soft', 0.25)
        self.assertEqual(asoft, a_ann)
        self.assertEqual(wasoft, [1.0, 0.25, 0.25, 1.0])

    def test_postprocess_spans_v3_merges_and_trims_punctuation(self):
        text = "Intime-se a parte, para manifestar;"
        spans = [
            dict(label='Encaminhamento', begin=0, end=10, char_start_index=0, char_end_index=10, span='Intime-se '),
            dict(label='Encaminhamento', begin=9, end=18, char_start_index=9, char_end_index=18, span=' a parte,'),
        ]
        out = postprocess_spans_v3(text, spans)
        self.assertEqual(len(out), 2)
        self.assertEqual((out[1]['begin'], out[1]['end'], out[1]['span']), (10, 17, 'a parte'))

    def test_predict_relations_v3_estabelecido_and_continuado(self):
        text = "Defiro o pedido. Intime-se no prazo de 15 dias para cumprir."
        entities = [
            dict(label='Pronunciamento', begin=0, end=15, span='Defiro o pedido'),
            dict(label='Encaminhamento', begin=17, end=26, span='Intime-se'),
            dict(label='Prazo', begin=30, end=46, span='prazo de 15 dias'),
            dict(label='Encaminhamento', begin=47, end=59, span='para cumprir'),
        ]
        rels = predict_relations_v3(text, entities)
        self.assertIn(dict(label='continuado', source=1, target=3), rels)
        self.assertIn(dict(label='estabelecido', source=2, target=1), rels)
        self.assertIn(dict(label='vinculado', source=1, target=0), rels)


if __name__=='__main__':unittest.main()
