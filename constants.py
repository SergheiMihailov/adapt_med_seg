import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")

DATASETS_TABLE = """
ID	Dataset	Link
0000	CHAOS	https://chaos.grand-challenge.org/
0001	HaN-Seg	https://han-seg2023.grand-challenge.org/
0002	AMOS22	https://amos22.grand-challenge.org/
0003	AbdomenCT-1k	https://github.com/JunMa11/AbdomenCT-1K
0004	KiTS23	https://kits-challenge.org/kits23/
0005	KiPA22	https://kipa22.grand-challenge.org/
0006	KiTS19	https://kits19.grand-challenge.org/
0007	BTCV	https://www.synapse.org/#!Synapse:syn3193805/wiki/217753
0008	Pancreas-CT	https://wiki.cancerimagingarchive.net/display/public/pancreas-ct
0009	3D-IRCADB	https://www.kaggle.com/datasets/nguyenhoainam27/3dircadb
0010	FLARE22	https://flare22.grand-challenge.org/
0011	TotalSegmentator	https://github.com/wasserth/TotalSegmentator
0012	CT-ORG	https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890
0013	WORD	https://paperswithcode.com/dataset/word
0014	VerSe19	https://osf.io/nqjyw/
0015	VerSe20	https://osf.io/t98fz/
0016	SLIVER07	https://sliver07.grand-challenge.org/
0017	QUBIQ	https://qubiq.grand-challenge.org/
0018	MSD-Colon	http://medicaldecathlon.com/
0019	MSD-HepaticVessel	http://medicaldecathlon.com/
0020	MSD-Liver	http://medicaldecathlon.com/
0021	MSD-lung	http://medicaldecathlon.com/
0022	MSD-pancreas	http://medicaldecathlon.com/
0023	MSD-spleen	http://medicaldecathlon.com/
0024	LUNA16	https://luna16.grand-challenge.org/Data/
"""

M3D_DATASET_DICT = {
    line.split("\t")[0]: line.split("\t")[1]
    for line in DATASETS_TABLE.strip().split("\n")[1:]
}
