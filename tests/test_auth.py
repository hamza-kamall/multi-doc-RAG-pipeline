def test_current_user(authenticated_client):
    response = authenticated_client.get("/auth/me")

    assert response.status_code == 200
    assert response.json()["username"] == "test_user"
    assert "password" not in response.json()
    assert "password_hash" not in response.json()


def test_login_rejects_wrong_password(client):
    registration = client.post(
        "/users",
        json={
            "username": "login_user",
            "password": "correct_password_123",
        },
    )
    assert registration.status_code == 201

    response = client.post(
        "/auth/login",
        data={
            "username": "login_user",
            "password": "wrong_password_123",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password"


def test_documents_require_authentication(client):
    response = client.get("/documents")

    assert response.status_code == 401
