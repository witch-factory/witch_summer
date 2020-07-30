from abc import ABC, abstractmethod
from collections import namedtuple

Customer=namedtuple('Customer', 'name fidelity')


class LineItem: #구입한 물건
    def __init__(self, product, quantity, price):
        self.product=product
        self.quantity=quantity
        self.price=price

    def total(self):
        return self.price*self.quantity


class Order:
    def __init__(self, customer, cart, promotion=None): #Customer 튜플 받음
        self.customer=customer
        self.cart=list(cart) #LineItem들이 들어 있음
        self.promotion=promotion #적용되는 할인. 기본 할인은 없다

    def total(self):
        if not hasattr(self, '__total'):
            self.__total=sum(item.total() for item in self.cart)
        return self.__total

    def due(self):
        if self.promotion is None:
            discount=0
        else:
            discount=self.promotion.discount(self) #깎이는 비용
        return self.total()-discount

    def __repr__(self):
        fmt='<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())


class Promotion(ABC): #abstract class inherit
    def discount(self, order): #할인액을 구체적 숫자로 변환
        """상속해서 오버라이딩하는 것이 목적인 추상 클래스. 각각의 조건에 따라 할인을 적용하게 된다"""


class FidelityPromo(Promotion): #충성도 1000 이상이면 5% 전체할인 적용
    def discount(self, order):
        return order.total()*0.05 if order.customer.fidelity>=1000 else 0


class BulkItemPromo(Promotion): #같은 상품 20개 이상 구입시 그 상품군 10% 할인 적용
    def discount(self, order):
        dc=0
        for item in order.cart:
            if item.quantity>=20:
                dc+=item.total()*0.1
        return dc


class LargeOrderPromo(Promotion): #10종류 이상 상품 구입시 7% 전체할인 적용
    def discount(self, order):
        distinct_items={item.product for item in order.cart} #set을 이용해 상품 종류 계산
        if len(distinct_items)>=10:
            return order.total*0.07
        return 0


Joe=Customer('John Doe',0)
Ann=Customer('Ann Smith',1100)
cart=[LineItem('banana', 4, 0.5),
      LineItem('apple', 25, 1.5),
      LineItem('melon', 5, 5.0)]
print(Order(Joe, cart, BulkItemPromo()))
print(Order(Ann, cart, FidelityPromo()))


