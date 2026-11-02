"""Which object each provided real scan shows, as told by the user (the box scans by elimination).

TEST LABELS ONLY: used to evaluate predictions on the real scans. Never used for training and never
shown as a prediction."""
TEST_LABELS = {
    'scan_20261015_132356_shapenet': 'bottle',  # red bottle
    'scan_20261015_132816_shapenet': 'bottle',  # red bottle
    'scan_20261015_133832_shapenet': 'bottle',  # red bottle
    'scan_20261015_134311_shapenet': 'bottle',  # red bottle
    'scan_20261015_135157_shapenet': 'bottle',  # blue bottle
    'scan_20261015_135527_shapenet': 'bottle',  # blue bottle
    'scan_20261015_140131_shapenet': 'bottle',  # blue bottle
    'scan_20261015_140908_shapenet': 'box',
    'scan_20261015_141027_shapenet': 'box',
    'scan_20261015_141202_shapenet': 'box',
}
USER_NAMES = {k: v for k, v in zip(TEST_LABELS, ['Red bottle'] * 4 + ['Blue bottle'] * 3 + ['Box'] * 3)}
