import pytest

from app.data_access.database import init_db, make_engine, make_session_factory
from app.domain.models import Customer


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        s.add(Customer(customer_id=1, full_name="Demo Customer", email="demo@example.com",
                       password="x", phone="000", address="Hanoi"))
        s.commit()
    yield factory
    engine.dispose()
