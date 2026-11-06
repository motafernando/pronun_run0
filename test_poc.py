import unittest
from poc import group_documents, encode_labels, decode_spans, class_weights, match_iou, continuation_multipliers, split_dataset, HEAD_A


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


if __name__=='__main__':unittest.main()
