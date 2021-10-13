from neo4j import GraphDatabase
import logging
from neo4j.exceptions import ServiceUnavailable
from utils import get_personName,eng2cn
from model.utils import save_parameter
class Neo4jDriver:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    def create_friendship(self, person1_name, person2_name):
        with self.driver.session() as session:
            # Write transactions allow the driver to handle retries and transient errors
            result = session.write_transaction(
                self._create_and_return_friendship, person1_name, person2_name)
            for row in result:
                print("Created friendship between: {p1}, {p2}".format(p1=row['p1'], p2=row['p2']))

    @staticmethod
    def _create_and_return_friendship(tx, person1_name, person2_name):
        query = (
            "CREATE (p1:Person { name: $person1_name }) "
            "CREATE (p2:Person { name: $person2_name }) "
            "CREATE (p1)-[:KNOWS]->(p2) "
            "RETURN p1, p2"
        )

        result = tx.run(query, person1_name=person1_name, person2_name=person2_name)
        try:
            return [{"p1": row["p1"]["name"], "p2": row["p2"]["name"]}
                    for row in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
    
    
    def touch_person_data(self,N):
        with self.driver.session() as session:
            result = session.read_transaction(self._touch_and_return_data, N)
            for row in result:
                print(row)

    @staticmethod
    def _touch_and_return_data(tx, N):
        query = (
            "MATCH (n) "
            "RETURN n "
            "LIMIT $N"
        )
        result = tx.run(query, N=N)
        return [ row for row in result]


    def from_person_to_company(self, person_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_person_to_company, person_name)
            msg_box = []
            for row in result:
                _, status = get_personName(row[1])
                msg = "{person_name}在{LStkNm}工作,担任{status}的职务".format(person_name=person_name,LStkNm=row[0]['LStkNm'],status=eng2cn(status))
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_person_to_company(tx, person_name):
        query=(
            "MATCH (p1)-[]->(n:company) "
            "WHERE p1.name_AffRepr=$person_name OR p1.name_Secbd=$person_name OR p1.name_genmgr=$person_name OR p1.name_legalRepr=$person_name "
            "RETURN p1,n"
        )
        result = tx.run(query, person_name=person_name)
        
        try:
            return [[item['n'],item['p1']] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise

    def from_industry_to_company(self, industry_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_industry_to_company, industry_name)
            msg_box = []
            for row in result:
                msg = "{comp_name}归属于{industry}".format(comp_name=row['LStkNm'],industry=industry_name)
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_industry_to_company(tx, industry_name):
        query=(
            "MATCH (n:company)-->(m:industry{name_industry:$industry_name}) "
            "RETURN n"
        )
        result = tx.run(query, industry_name=industry_name)
        try:
            return [item['n'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
    

    def from_location_match_industry_for_company(self, location_name, industry_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_location_match_industry_for_company, location_name, industry_name)
            msg_box = []
            for row in result:
                msg = "{name}是在{location}的{industry}公司".format(name=row['LStkNm'],location=location_name,industry=industry_name)
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_location_match_industry_for_company(tx, location_name,industry_name):
        query=(
            "MATCH (k:industry{name_industry:$industry_name})<--(m:company)-->(n:prov{name_prov:$location_name}) "
            "RETURN m"
        )
        result = tx.run(query, industry_name=industry_name, location_name=location_name)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise

    def from_company_match_relative(self, company_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_company_match_relative, company_name)
            msg_box = []
            for row in result:
                msg = "{comp_name}也属于{location}".format(comp_name=row['LStkNm'],location=row['CsrcIcNm1'])
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_company_match_relative(tx, company_name):
        query=(
            "MATCH (:company{LStkNm:$company_name})-->(:industry)<--(n:company) "
            "RETURN n"
        )
        result = tx.run(query, company_name=company_name)
        try:
            return [item['n'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise

    def from_location_to_company(self, location_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_location_to_company, location_name)
            msg_box = []
            for row in result:
                msg = "{comp_name}在{location}".format(comp_name=row['LStkNm'],location=location_name)
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_location_to_company(tx, location_name):
        query=(
            "MATCH (n:company)-->(m:prov{name_prov:$location_name}) "
            "RETURN n"
        )
        result = tx.run(query, location_name=location_name)
        try:
            return [item['n'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
    
    def from_company_to_AffRepr(self, company_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_company_to_AffRepr, company_name)
            msg_box = []
            for row in result:
                print(row)
                msg = "{comp_name}的证券事务代表是{secbd}".format(comp_name=company_name,secbd=row['name_AffRepr'])
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_company_to_AffRepr(tx, company_name):
        query=(
            "MATCH (m:Aff_Repr)-->(n:company{LStkNm:$company_name}) "
            "RETURN m"
        )
        result = tx.run(query, company_name=company_name)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
    
    def from_company_to_Secbd(self, company_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_company_to_Secbd, company_name)
            msg_box = []
            for row in result:
                print(row)
                msg = "{comp_name}的董事会秘书是{name}".format(comp_name=company_name,name=row['name_Secbd'])
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_company_to_Secbd(tx, company_name):
        query=(
            "MATCH (m:Secbd)-->(n:company{LStkNm:$company_name}) "
            "RETURN m"
        )
        result = tx.run(query, company_name=company_name)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
    
    def from_company_to_legal(self, company_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_company_to_legal, company_name)
            msg_box = []
            for row in result:
                print(row)
                msg = "{comp_name}的法人代表是{name}".format(comp_name=company_name,name=row['name_legalRepr'])
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_company_to_legal(tx, company_name):
        query=(
            "MATCH (m:legal_Repr)-->(n:company{LStkNm:$company_name}) "
            "RETURN m"
        )
        result = tx.run(query, company_name=company_name)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
    
    def from_company_to_genmgr(self, company_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_company_to_genmgr, company_name)
            msg_box = []
            for row in result:
                print(row)
                msg = "{comp_name}的总经理是{name}".format(comp_name=company_name,name=row['name_genmgr'])
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_company_to_genmgr(tx, company_name):
        query=(
            "MATCH (m:genmgr)-->(n:company{LStkNm:$company_name}) "
            "RETURN m"
        )
        result = tx.run(query, company_name=company_name)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise

    def from_company_to_allmgr(self, company_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_company_to_allmgr, company_name)
            name_list = ['name_genmgr','name_legalRepr','name_AffRepr','name_Secbd']
            msg_box = []
            for item in result:  
                name,status = get_personName(item)
                msg = '{status}:{person_name}'.format(status=eng2cn(status),person_name=name)
                msg_box.append(msg)
                print(msg)
            return msg_box,result

    @staticmethod
    def _from_company_to_allmgr(tx, company_name):
        query=(
            "MATCH (m:genmgr)-->(n:company{LStkNm:$company_name}) "
            "RETURN m"
        )
        query=("MATCH (m)-[r:`证券事务代表`|`总经理`|`法人代表|董事会秘书`]->(n:company{LStkNm:$company_name})"
        "RETURN m")
        result = tx.run(query, company_name=company_name)
        
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise


    def from_company_to_position(self, company_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_company_to_position, company_name)
            msg_box = []
            for row in result:
                print(row)
                msg = "{comp_name}位于{name}".format(comp_name=company_name,name=row['name_prov'])
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_company_to_position(tx, company_name):
        query=(
            "MATCH (n:company{LStkNm:$company_name})-->(m:prov) "
            "RETURN m"
        )
        result = tx.run(query, company_name=company_name)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
    
    def from_company_to_industry(self, company_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_company_to_industry, company_name)
            msg_box = []
            for row in result:
                print(row)
                msg = "{comp_name}归属于{name}".format(comp_name=company_name,name=row['name_industry'])
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_company_to_industry(tx, company_name):
        query=(
            "MATCH (n:company{LStkNm:$company_name})-->(m:industry) "
            "RETURN m"
        )
        result = tx.run(query, company_name=company_name)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
    
    def from_company_query_business(self, company_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._from_company_query_business, company_name)
            msg_box = []
            for row in result:
                msg = "{comp_name}的主营业务{name}".format(comp_name=company_name,name=row['MainBusiness'])
                msg_box.append(msg)
                print(msg)
            return msg_box, result

    @staticmethod
    def _from_company_query_business(tx, company_name):
        query=(
            "MATCH (m:company{LStkNm:$company_name}) "
            "RETURN m"
        )
        result = tx.run(query, company_name=company_name)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise

    def acquire_all_person(self):
        with self.driver.session() as session:
            result = session.read_transaction(self._acquire_all_person)
            msg_box = [get_personName(row)[0] for row in result]
            return msg_box, result

    @staticmethod
    def _acquire_all_person(tx):
        query=(
            "MATCH (m)-[r:`证券事务代表`|`总经理`|`法人代表`|`董事会秘书`]->(n:company) "
            "RETURN m"
        )
        result = tx.run(query)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise

    @staticmethod
    def _acquire_all_company(tx):
        query=(
            "MATCH (m:company) "
            "RETURN m"
        )
        result = tx.run(query)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
        
    def acquire_all_company(self):
        with self.driver.session() as session:
            result = session.read_transaction(self._acquire_all_company)
            msg_box = [row['LStkNm'] for row in result]
            return msg_box, result

    @staticmethod
    def _acquire_all_province(tx):
        query=(
            "MATCH (m:prov) "
            "RETURN m"
        )
        result = tx.run(query)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
        
    def acquire_all_province(self):
        with self.driver.session() as session:
            result = session.read_transaction(self._acquire_all_province)
            msg_box = [row['name_prov'] for row in result]
            return msg_box, result

    @staticmethod
    def _acquire_all_industry(tx):
        query=(
            "MATCH (m:industry) "
            "RETURN m"
        )
        result = tx.run(query)
        try:
            return [item['m'] for item in result]
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise
        
    def acquire_all_industry(self):
        with self.driver.session() as session:
            result = session.read_transaction(self._acquire_all_industry)
            msg_box = [row['name_industry'] for row in result]
            return msg_box, result

if __name__ == "__main__":
    # Aura queries use an encrypted connection using the "neo4j+s" URI scheme
    bolt_url = "bolt://localhost:7687"
    user = "neo4j"
    password = "Neo4j7474$"
    Neo4jDriver = Neo4jDriver(bolt_url, user, password)
    #Neo4jDriver.create_friendship("Alice", "David")
    #Neo4jDriver.touch_person_data(5)
    
    msg_box,_ = Neo4jDriver.acquire_all_person()
    for item in msg_box:
        if item == '张丹石':
            print(1)
    PATH = '.\src\model\ckpts\person_name.pkl'
    save_parameter(msg_box,PATH)
    
    #Neo4jDriver.from_company_match_relative("西藏天路")
    #Neo4jDriver.from_company_to_allmgr("白云山")
    #Neo4jDriver.from_company_query_business("白云山")
    
    Neo4jDriver.close()
