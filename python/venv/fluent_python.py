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
            discount=self.promotion(self) #깎이는 비용
        return self.total()-discount

    def __repr__(self):
        fmt='<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())


promos=[]


def promotion(promo_func):
    promos.append(promo_func)
    return promo_func


@promotion
def fidelity_promo(order):
    return order.total()*0.05 if order.customer.fidelity>=1000 else 0


@promotion
def bulkitem_promo(order):
    dc=0
    for item in order.cart:
      if item.quantity>=20:
          dc+=item.total()*0.1
    return dc


@promotion
def largeorder_promo(self, order):
    distinct_items={item.product for item in order.cart} #set을 이용해 상품 종류 계산
    if len(distinct_items)>=10:
        return order.total*0.07
    return 0


def best_promo(order):
    return max(promo(order) for promo in promos)